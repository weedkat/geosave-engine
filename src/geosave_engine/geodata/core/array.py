"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.attrs.headers.geobox import (
    create_header as create_geobox_header,
)
from geosave_engine.geodata.transform import nodata, packing, warp
from geosave_engine.geodata.utils.io.geotiff import write_cog, write_gtiff

from .base import GeoAccessor, tensor
from .profile import (
    BAND_DIMENSION,
    CRS_COORDINATE,
    NOT_GEOREFERENCED_DIMENSIONS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from os import PathLike
    from pathlib import Path

    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike
    from odc.geo import SomeResolution

    from geosave_engine.geodata.utils.io.geotiff import (
        COGWriteOptions,
        GTiffWriteOptions,
    )

    from geosave_engine.geodata import DataArray, Dataset
    from geosave_engine.geodata.transform.warp import Resampling


def array(
    pixels: np.ndarray,
    geobox: GeoBox | None = None,
    /,
    *,
    nodata: float | int | None = None,
    **coords: Sequence[Any] | np.ndarray | None,
) -> DataArray:
    """Build one band from an array on a grid.

    The array ends in the two spatial axes, which the geobox names, and lies
    along the axes `coords` names ahead of them. Build several bands at
    once with `raster`, which returns them as one Dataset.

    Args:
        pixels: Values, ending in the two spatial axes.
        geobox: Grid placing the trailing two axes. None leaves the band
            unreferenced, so it holds pixels without claiming ground position.
        nodata: Value standing for nodata pixels, written as
            `Nodata.fill_value`. None writes no fill value.
        **coords: Leading axis name mapped to its labels, in array order. None
            labels an axis carrying none. Pass a name Python reserves as
            `**{"class": labels}`.

    Returns:
        Georeferenced DataArray when `geobox` is given, its spatial
        coordinates naming what they measure, otherwise a DataArray carrying
        no grid, whose spatial dims are named `y` and `x`.

    Raises:
        ValueError: `pixels` rank does not match `coords` plus the spatial
            pair, its trailing axes do not match `geobox`, an axis is labelled
            with the wrong number of values, or an axis name collides with a
            coordinate the grid supplies.

    Examples:
        >>> array(logits, geobox, **{"class": ["water", "urban", "crop"]})
        <xarray.DataArray (class: 3, y: 512, x: 512)> Size: 3MB
        Coordinates:
          * class        (class) <U5 'water' 'urban' 'crop'
          * y            (y) float64 5.005e+06 5.005e+06 ... 5e+06
          * x            (x) float64 3e+05 3e+05 ... 3.051e+05
            spatial_ref  int32 32633
    """
    from odc.geo.xr import wrap_xr

    spatial_dims = NOT_GEOREFERENCED_DIMENSIONS if geobox is None else geobox.dimensions
    grid_coords = (*spatial_dims, CRS_COORDINATE)
    band_axes = tuple(coords)

    collisions = sorted(set(band_axes) & set(grid_coords))
    if collisions:
        raise ValueError(
            f"{collisions} name coordinates this grid already supplies "
            f"{list(grid_coords)}; name the leading axes something else"
        )

    if pixels.ndim != len(band_axes) + 2:
        raise ValueError(
            f"pixels are {pixels.ndim}-dimensional but lie along {band_axes} ahead "
            f"of the spatial pair; a band's array ends in the two spatial axes"
        )
    if geobox is not None and pixels.shape[len(band_axes) :] != geobox.shape:
        raise ValueError(
            f"the pixels' trailing axes are {pixels.shape[len(band_axes) :]} but the "
            f"geobox is {tuple(geobox.shape)}; place them on a matching grid first"
        )

    # coords is in array order, so a keyword's position is its axis's position.
    miscounted = []
    for position, (axis, labels) in enumerate(coords.items()):
        if labels is None:
            continue
        length = pixels.shape[position]
        if len(labels) != length:
            miscounted.append(
                f"{axis!r} got {len(labels)} labels for an axis of {length}"
            )
    if miscounted:
        raise ValueError(
            f"{'; '.join(miscounted)}; label an axis once per value along it"
        )

    band_dims = (*band_axes, *spatial_dims)
    if geobox is None:
        built = xr.DataArray(pixels, dims=band_dims)
    else:
        built = wrap_xr(pixels, geobox, dims=band_dims, axis=len(band_axes))

    axis_labels = {axis: given for axis, given in coords.items() if given is not None}
    if axis_labels:
        built = built.assign_coords(axis_labels)

    if nodata is not None:
        built = attrs.rebase(built, attrs.Nodata(fill_value=nodata))

    if geobox is not None:
        # odc places the axes; GeoSave writes what they measure.
        for name, semantics in create_geobox_header(geobox).coords.items():
            built = attrs.rebase(built, semantics, target=name)

    return cast("DataArray", built)


@xr.register_dataarray_accessor("gs")
class GeoArray(GeoAccessor["DataArray"]):
    """Read and transform one raster DataArray.

    A DataArray is the unit a panel draws, so this accessor carries what
    drawing needs and leaves whole-Dataset operations to `GeoRaster`.

    Args:
        data: DataArray holding one band.

    Examples:
        >>> ds["ndvi"].gs.plot(cmap="RdYlGn")
    """

    def __init__(self, data: xr.DataArray) -> None:
        """Bind the DataArray.

        Args:
            data: DataArray to read through this accessor.
        """
        self._data = cast("DataArray", data)

    @property
    def variables(self) -> tuple[str, ...]:
        """Name this array as a data variable, when it has a name."""
        return () if self._data.name is None else (str(self._data.name),)

    @property
    def grid_dims(self) -> tuple[str, str]:
        """Name the two dimensions the grid spans.

        Returns:
            `("y", "x")` for a projected CRS, `("latitude", "longitude")` for
            a geographic one, and `("y", "x")` for a band carrying no grid,
            whose pixels span those axes without claiming ground position.
        """
        grid = self._data.odc.geobox
        return NOT_GEOREFERENCED_DIMENSIONS if grid is None else grid.dimensions

    @property
    def axes(self) -> dict[str, np.ndarray | None]:
        """Read the axes this band spans besides the grid.

        Returns:
            {
                "<axis name>": its labels, or None where it carries none,
            }
            In the band's own order, ready to pass to `array` or `raster`.

        Examples:
            >>> ds["ndvi"].gs.axes
            {'time': array(['2025-06-01T00:00:00.000000000'], dtype='datetime64[ns]')}
        """
        grid_dims = self.grid_dims
        return {
            str(dim): self._data.coords[dim].values
            if dim in self._data.coords
            else None
            for dim in self._data.dims
            if dim not in grid_dims
        }

    def write_nodata(self, value: float | int | None) -> DataArray:
        """Write the stored value standing for this band's nodata pixels.

        `Nodata` mirrors the value across both `_FillValue` and odc's own
        `nodata`, so no reader can find the two naming different pixels.

        Args:
            value: Stored value marking nodata, of this band's own type. None
                clears it.

        Returns:
            New DataArray carrying `value` as its fill.

        Raises:
            ValueError: `value` is not of this band's type.

        Examples:
            >>> ds["red"].gs.write_nodata(0).attrs["_FillValue"]
            0
        """
        if value is not None:
            nodata.check_fill_fits(value, self._data.dtype, str(self._data.name))
        return attrs.rebase(self._data, attrs.Nodata(fill_value=value))

    def unpack(self) -> DataArray:
        """Read physical values out of this band's stored digital numbers.

        Returns:
            New DataArray of physical values, or this band unchanged where it
            carries neither `scale_factor` nor `add_offset`.

        Examples:
            >>> ds["B04"].gs.unpack().max().item()
            0.09
        """
        return packing.unpack(self._data)

    def mask(
        self, valid: xr.DataArray | np.ndarray, *, fill: float | int | None = None
    ) -> DataArray:
        """Make nodata every pixel `valid` does not keep.

        Args:
            valid: Boolean array, True where a pixel is real data. A
                DataArray names its own axes and may span fewer than this
                band, broadcasting over the rest; a bare numpy array names
                none, so it is read as the grid alone and must match its
                shape.
            fill: Value the blanked pixels take, also written onto the result
                if this band carries no fill value yet. None reads what it
                already carries.

        Returns:
            New DataArray, its unkept pixels holding the fill value it
            carries.

        Raises:
            ValueError: `valid` is not boolean, does not span the grid, spans
                an axis this band does not, `fill` does not fit its dtype, or
                it carries no fill value and none is given.

        Examples:
            >>> clear = ds.scl.gs.mask(ds.scl.isin([4, 5, 6, 7]))
        """
        return nodata.mask(self._data, valid, fill=fill)

    def to_nan(self) -> DataArray:
        """Replace this band's fill value with NaN.

        Returns:
            New DataArray holding NaN where the pixels were nodata, or this
            band unchanged where it carries no fill value.

        Examples:
            >>> ds.red.gs.to_nan().dtype
            dtype('float64')
        """
        return nodata.to_nan(self._data)

    def reproject(
        self,
        target: warp.Target,
        *,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | None = None,
    ) -> DataArray:
        """Warp pixels onto the grid a target names.

        A target grid is adopted whole — CRS, resolution, and extent — while a
        bare CRS only decides the projection, sizing the grid from this band.

        Args:
            target: Grid to land on, a raster already on one, or a CRS.
            resampling: One GDAL kernel for every variable, or a mapping
                naming each variable's own, which `"*"` answers the rest of.
            resolution: Output pixel size, taken only for a CRS target. None
                keeps this band's ground sampling as closely as the new CRS
                allows.

        Returns:
            New DataArray on the target grid, carrying its own `spatial_ref`
            and the CF semantics its axes earn.

        Raises:
            ValueError: This band or `target` sits on no locatable grid,
                `resolution` contradicts a target grid, or `resampling` would
                blend a band whose values are class codes.

        Examples:
            >>> ds.red.gs.reproject("EPSG:3857").gs.crs.epsg
            3857
            >>> dem = srtm.gs.reproject(scene.red, resampling="bilinear")
            >>> dem.gs.geobox == scene.gs.geobox
            True
        """
        return warp.reproject(
            self._data, target, resampling=resampling, resolution=resolution
        )

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> np.ndarray:
        """Read this band as one model-input array.

        The grid pair trails and a `band` axis sits just ahead of it, matching
        what `GeoRaster.to_numpy` builds by stacking variables. Other axes keep
        this band's own order.

        Args:
            dtype: Cast applied to the pixels. None keeps their own dtype.

        Returns:
            Array shaped `(*axes, band, y, x)`, or `(*axes, y, x)` where this
            band spans no `band` axis.

        Examples:
            >>> ds["ndvi"].gs.to_numpy().shape
            (2, 256, 256)
        """
        # Put spatial dims (y, x) last
        ahead = [axis for axis in self.axes if axis != BAND_DIMENSION]
        if BAND_DIMENSION in self._data.dims:
            ahead.append(BAND_DIMENSION)

        ordered = self._data.transpose(*ahead, *self.grid_dims).values
        return ordered if dtype is None else ordered.astype(dtype)

    def to_tensor(
        self, *, dtype: str | torch.dtype | None = None
    ) -> torch.Tensor:
        """Read this band as one model-input tensor.

        Args:
            dtype: Torch dtype or its YAML-friendly name. None preserves the
                prepared array dtype.

        Returns:
            Tensor shaped `(*axes, band, y, x)`, or `(*axes, y, x)` where this
            band spans no `band` axis.

        Examples:
            >>> ds["ndvi"].gs.to_tensor().dtype
            torch.uint16
        """
        return tensor(lambda reading: self.to_numpy(dtype=reading), dtype)

    def to_cog(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        **options: Unpack[COGWriteOptions],
    ) -> Path:
        """Write this band as a Cloud Optimized GeoTIFF.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            **options: COG creation options.

        Returns:
            The written path.

        Raises:
            ValueError: This array names none of the bands it spans, carries
                no `.name` to write a sole band under, or carries a `time`
                dimension rather than a scalar coordinate.
            FileExistsError: `path` exists and `overwrite` is false.

        Examples:
            >>> ds["ndvi"].gs.to_cog("ndvi.tif")
            PosixPath('ndvi.tif')
        """
        return write_cog(
            self.to_raster(),
            path,
            map_scale=map_scale,
            overwrite=overwrite,
            **options,
        )

    def to_gtiff(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        **options: Unpack[GTiffWriteOptions],
    ) -> Path:
        """Write this band as a plain GeoTIFF.

        Reach for `to_cog` unless a consumer needs a striped or otherwise
        non-COG file.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            **options: GTiff creation options.

        Returns:
            The written path.

        Raises:
            ValueError: This array names none of the bands it spans, carries
                no `.name` to write a sole band under, or carries a `time`
                dimension rather than a scalar coordinate.
            FileExistsError: `path` exists and `overwrite` is false.
        """
        return write_gtiff(
            self.to_raster(),
            path,
            map_scale=map_scale,
            overwrite=overwrite,
            **options,
        )

    def to_raster(self) -> Dataset:
        """Split this array back into the raster whose variables it stacks.

        The inverse of `GeoRaster.to_array`: each band becomes one variable,
        carrying the attrs that ride on the `band` coordinate. An array
        spanning no band axis becomes a one-variable raster named after it.

        Returns:
            Dataset holding one variable per band. The array's own attrs
            become the raster's, less the ones its bands take back.

        Raises:
            ValueError: This array spans a `band` axis carrying no labels to
                name its variables, or spans none and carries no `.name`.

        Examples:
            >>> ds.gs.to_array().gs.to_raster()
            >>> ds["ndvi"].gs.to_raster()
        """
        if BAND_DIMENSION not in self._data.dims:
            if self._data.name is None:
                raise ValueError("array carries no name to write its band under")
            return cast("Dataset", self._data.to_dataset())

        if BAND_DIMENSION not in self._data.coords:
            raise ValueError(
                f"this array spans {self._data.sizes[BAND_DIMENSION]} bands but "
                f"labels none of them, so they name no variables; assign a "
                f"{BAND_DIMENSION!r} coordinate first"
            )

        parked = self.attrs.coords[BAND_DIMENSION].get(attrs.StackedAttrs)
        raster = cast("Dataset", self._data.to_dataset(dim=BAND_DIMENSION))
        if parked is None:
            return raster

        restored = parked.restore({str(name) for name in raster.data_vars})
        for name, held in restored.items():
            raster[name].attrs = held
        raster.attrs = parked.restore_root(raster.attrs)
        return raster

    def statistics(self) -> BandSummary:
        """Summarise this band's present pixels, reading every one of them.

        A pixel is absent where it is NaN or holds the band's own fill value,
        which is what GDAL summarises too. A chunked band is computed, so this
        costs a full read.

        Returns:
            Summary of the pixels this band calls present.

        Raises:
            ValueError: Every pixel is absent, so there is nothing to
                summarise.

        Examples:
            >>> ds["B04"].gs.statistics()
            BandSummary(minimum=1.0, maximum=63.0, mean=32.0, stddev=18.2, valid_percent=98.4)
        """
        fill = self.attrs.root.get(attrs.Nodata)
        values = (
            self._data
            if fill is None or fill.fill_value is None
            else self._data.where(self._data != fill.fill_value)
        )

        summary = xr.Dataset(
            {
                "present": values.notnull().sum(),
                "minimum": values.min(),
                "maximum": values.max(),
                "mean": values.mean(),
                "stddev": values.std(),
            }
        ).compute()
        present = int(summary["present"])
        if not present:
            raise ValueError(
                f"{self._data.name} holds no present pixel, so it summarises to "
                f"nothing; drop the band or give it pixels that are not fill"
            )
        return BandSummary(
            minimum=float(summary["minimum"]),
            maximum=float(summary["maximum"]),
            mean=float(summary["mean"]),
            stddev=float(summary["stddev"]),
            valid_percent=100.0 * present / values.size,
        )

    def colorize(self) -> DataArray:
        """Bake the class colours this band carries into display channels.

        Reads `Legend`, colouring each pixel by the class its code names. The
        result holds colour rather than codes, so it draws without a legend;
        prefer `plot` where a named colorbar is wanted.

        Returns:
            Georeferenced array shaped `(*axes, band, y, x)` valued in
            `[0, 1]`, whose `band` coordinate is `("red", "green", "blue")`.
            A pixel no class names is absent on every channel.

        Raises:
            ValueError: The band lists no classes, or names a class carrying
                no colour.

        Examples:
            >>> ds["landcover"].gs.colorize().sizes["band"]
            3
        """
        from geosave_engine.utils.colorize import parse_color

        legend = self.attrs.root.get(attrs.Legend)
        class_map = legend.class_map if legend is not None else None
        if class_map is None:
            raise ValueError(
                "band lists no classes, so its values name none to colour; write "
                "a Legend, or compose channels with GeoRaster.to_array"
            )

        colour_of = legend.color_map if legend is not None else None
        codes = sorted(class_map)
        missing_colour = [code for code in codes if code not in (colour_of or {})]
        if missing_colour:
            raise ValueError(
                f"classes {missing_colour} carry no colour; give Legend.color_map an "
                f"entry for every class the band lists"
            )
        colour_of = colour_of or {}

        palette = np.array(
            [parse_color(colour_of[code]) for code in codes], dtype="float32"
        )
        palette /= 255.0  # (class, 3)

        pixels = self._data.values
        names_class = np.isin(pixels, codes)
        code_index = np.where(names_class, np.searchsorted(codes, pixels), 0)
        channels = np.where(names_class[..., None], palette[code_index], np.nan)

        return array(
            np.moveaxis(channels, -1, -3),  # (*axes, band, y, x)
            self._data.odc.geobox,
            nodata=None,  # absence is NaN here, which no fill value stands for
            **{**self.axes, BAND_DIMENSION: ["red", "green", "blue"]},
        )

    def plot(
        self,
        *,
        cmap: str | Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        cols: int = 4,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> hv.Element | hv.Layout:
        """Draw this band as an Image. Needs the `viz` extra.

        Draws the values as they stand, so decode a packed band first. A band
        listing classes in `Legend` draws through that palette with them
        named on the colorbar, while `viz.continuous` draws the bare codes.

        Args:
            cmap: Colormap. Refused for a band carrying a class map, whose
                palette is `Legend.color_map`.
            clim: Colour limits. Refused for a band carrying a class map,
                whose limits follow the class count.
            cols: Columns a `time` axis lays panels into. Ignored otherwise.
            title: Panel title above the axes. None suppresses automatic
                titles.
            xlabel: Location caption below the axes. None uses the band's
                reverse-geocoded location; an empty string suppresses it.

        Returns:
            Image over this band's grid, or a Layout of one panel per `time`
            value, `cols` wide.

        Raises:
            ValueError: The band carries no locatable grid, `cmap` or `clim`
                is given for a band carrying a class map, or its attrs
                contradict each other.

        Examples:
            >>> ds["ndvi"].gs.plot(cmap="RdYlGn")
            >>> ds["ndvi"].gs.plot(cols=6)  # one panel per bucket
        """
        from geosave_engine.geodata.viz import plot

        flags = self.attrs.root.get(attrs.Legend)
        legend = self.attrs.root.get(attrs.Legend)
        place = self.anchor.location if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return plot(
            self._data,
            cmap=cmap,
            clim=clim,
            class_map=flags.class_map if flags else None,
            color_map=legend.color_map if legend else None,
            cols=cols,
            title=title,
            xlabel=caption,
        )


class BandSummary(NamedTuple):
    """What one band's present pixels amount to.

    Args:
        minimum: Smallest value among the present pixels.
        maximum: Largest value among them.
        mean: Their arithmetic mean.
        stddev: Their population standard deviation.
        valid_percent: Share of the band's pixels that are present, as a
            percentage.

    Examples:
        >>> ds["B04"].gs.statistics()
        BandSummary(minimum=1.0, maximum=63.0, mean=32.0, stddev=18.2, valid_percent=98.4)
    """

    minimum: float
    maximum: float
    mean: float
    stddev: float
    valid_percent: float
