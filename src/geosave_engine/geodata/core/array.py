"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import nodata, packing, warp

from .base import GeoAccessor, tensor
from .convention import (
    BAND_DIMENSION,
    CRS_COORDINATE,
    NOT_GEOREFERENCED_DIMENSIONS,
    TIME_COORDINATE,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike
    from odc.geo import SomeCRS, SomeResolution

    from geosave_engine.geodata.utils.datetime import DateRange

    from geosave_engine.geodata import DataArray
    from geosave_engine.geodata.transform.warp import Resampling

    from .anchor import GeoAnchor


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

    shadowed = sorted(set(band_axes) & set(grid_coords))
    if shadowed:
        raise ValueError(
            f"{shadowed} name coordinates this grid already supplies "
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

    miscounted = [
        f"{axis!r} got {len(labels)} labels for an axis of {pixels.shape[position]}"
        for position, (axis, labels) in enumerate(coords.items())
        if labels is not None and len(labels) != pixels.shape[position]
    ]
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
        built = attrs.rebase(built, attrs.Nodata(_FillValue=nodata))

    if geobox is not None:
        # odc places the axes; GeoSave writes what they measure.
        for name, semantics in attrs.CFCoordinate.from_geobox(geobox).items():
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

    @property
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage.

        Returns:
            First and last covered instant, or None for timeless data.
        """
        if TIME_COORDINATE not in self._data.coords:
            return None
        spec = self.attrs.variables[TIME_COORDINATE].get(attrs.TimeSpec)
        labels = self._data.coords[TIME_COORDINATE].values
        return (spec or attrs.TimeSpec.instants()).timespan(labels)

    @property
    def anchor(self) -> GeoAnchor:
        """Read exact spatial and temporal coverage.

        Returns:
            Anchor over this band's grid and time span, which names its
            centroid, filename stem, and place.

        Raises:
            ValueError: The band carries no locatable grid.

        Examples:
            >>> ds["ndvi"].gs.anchor.stem
            '13.0016E_45.0011N_5.12kmx5.12km_10m'
        """
        from .anchor import GeoAnchor

        return GeoAnchor(self.geobox, timespan=self.timespan)

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
        # astype wraps, so an unheld -1 would name the pixel 65535 on uint16.
        if value is not None:
            with np.errstate(invalid="ignore"):
                stored = np.asarray(value).astype(self._data.dtype)
            if not np.array_equal(stored, np.asarray(value), equal_nan=True):
                raise ValueError(
                    f"{value!r} marks no pixel of {str(self._data.name)!r}, which "
                    f"is {self._data.dtype} and stores it as {stored.item()!r}; CF "
                    f"asks a fill value to have the variable's own type"
                )
        return attrs.rebase(self._data, attrs.Nodata(_FillValue=value))

    def unpack(self) -> DataArray:
        """Read physical values out of this band's stored digital numbers.

        Returns:
            New DataArray of physical values, or this band unchanged where it
            carries neither `scale_factor` nor `add_offset`.

        Examples:
            >>> ds["B04"].gs.unpack().max().item()
            0.09
        """
        return packing.decode(self._data)

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

    def decode(self) -> DataArray:
        """Replace this band's fill value with NaN.

        Returns:
            New DataArray holding NaN where the pixels were nodata, or this
            band unchanged where it carries no fill value.

        Examples:
            >>> ds.red.gs.decode().dtype
            dtype('float64')
        """
        return nodata.decode(self._data)

    def reproject(
        self,
        crs: SomeCRS,
        *,
        resampling: Resampling = "nearest",
        resolution: SomeResolution | None = None,
    ) -> DataArray:
        """Warp pixels into another CRS, deriving the grid to land on.

        Reach for `reproject_match` where the target grid already exists;
        this is for when only the CRS is decided and the grid is sized from
        the source.

        Args:
            crs: Coordinate reference system to land in.
            resampling: GDAL resampling kernel.
            resolution: Output pixel size. None keeps the source's ground
                sampling as closely as the new CRS allows.

        Returns:
            New DataArray in `crs`, on a grid covering the source.

        Raises:
            ValueError: This band carries no locatable grid, or `resampling`
                would blend a band whose values are class codes.

        Examples:
            >>> ds.red.gs.reproject("EPSG:3857").gs.crs.epsg
            3857
        """
        return warp.reproject(
            self._data, crs, resampling=resampling, resolution=resolution
        )

    def reproject_match(
        self,
        match: GeoBox | xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        resampling: Resampling = "nearest",
    ) -> DataArray:
        """Warp pixels onto a grid that already exists.

        Args:
            match: Grid to land on, or any xarray object carrying one. Its
                CRS, resolution, and extent are all adopted, so no resolution
                is taken.
            resampling: GDAL resampling kernel.

        Returns:
            New DataArray on the matched grid, carrying its own `spatial_ref`
            and the CF semantics its axes earn.

        Raises:
            ValueError: This band carries no locatable grid, or `resampling`
                would blend a band whose values are class codes.

        Examples:
            >>> dem = srtm.gs.reproject_match(scene.red, resampling="bilinear")
            >>> dem.gs.geobox == scene.gs.geobox
            True
        """
        return warp.reproject_match(self._data, match, resampling=resampling)

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

    def to_tensor(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Read this band as one model-input tensor.

        Args:
            dtype: Tensor dtype, which the pixels are also read in. None casts
                to `torch.float32`, which keeps unsigned imagery off
                `torch.uint16` — a dtype torch accepts and carries no
                arithmetic kernels for.

        Returns:
            Tensor shaped `(*axes, band, y, x)`, or `(*axes, y, x)` where this
            band spans no `band` axis.

        Examples:
            >>> ds["ndvi"].gs.to_tensor().dtype
            torch.float32
        """
        return tensor(lambda reading: self.to_numpy(dtype=reading), dtype)

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
            ValueError: The band carries no `Legend.class_map`, or names a
                class carrying no colour.

        Examples:
            >>> ds["landcover"].gs.colorize().sizes["band"]
            3
        """
        from geosave_engine.utils.colorize import parse_color

        legend = self.attrs.root.get(attrs.Legend)
        if not isinstance(legend, attrs.Legend) or legend.class_map is None:
            raise ValueError(
                "band carries no Legend.class_map, so its values name no classes "
                "to colour; write one, or compose channels with GeoRaster.to_array"
            )

        codes = sorted(legend.class_map)
        colour_of = legend.color_map or {}
        missing_colour = [code for code in codes if code not in colour_of]
        if missing_colour:
            raise ValueError(
                f"classes {missing_colour} carry no colour; give Legend.color_map an "
                f"entry for every class in class_map"
            )

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
        carrying `Legend.class_map` draws through that palette with its classes
        named on the colorbar, while `viz.continuous` draws the bare codes.

        Args:
            cmap: Colormap. Refused for a band carrying a class map, whose
                palette is `Legend.color_map`.
            clim: Colour limits. Refused for a band carrying a class map,
                whose limits follow the class count.
            cols: Columns a `time` axis lays panels into. Ignored otherwise.
            title: Panel title, above the axes. None leaves hvplot's own.
            xlabel: Caption below the axes, e.g. a place name. None leaves
                hvplot's own.

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

        legend = self.attrs.root.get(attrs.Legend)
        return plot(
            self._data,
            cmap=cmap,
            clim=clim,
            class_map=legend.class_map if legend else None,
            color_map=legend.color_map if legend else None,
            cols=cols,
            title=title,
            xlabel=xlabel,
        )
