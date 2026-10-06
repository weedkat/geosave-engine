"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import nodata

from .base import GeoRasterAccessor
from geosave_engine.geodata.conventions import (
    BAND_DIMENSION,
    NOT_GEOREFERENCED_DIMENSIONS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike
    from pathlib import Path

    import holoviews as hv
    import pystac
    from numpy.typing import DTypeLike

    from geosave_engine.geodata.io.storage import StorageOptions
    from geosave_engine.geodata.io.geotiff import (
        COGWriteOptions,
        GTiffWriteOptions,
    )

    from geosave_engine.geodata import DataArray, Dataset


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
    along the axes `coords` names ahead of them. It is `raster` with one
    variable, which builds several bands at once as one Dataset.

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
    from .raster import raster

    # A band is a one-variable raster, so it is built and checked the same way.
    built = raster({"pixels": pixels}, geobox, nodata=nodata, **coords)
    return cast("DataArray", built["pixels"].rename(None))


@xr.register_dataarray_accessor("gs")
class GeoArray(GeoRasterAccessor["DataArray"]):
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

    @property
    def nodata(self) -> float | int | None:
        """Read the stored value standing for nodata, or None where none is set."""
        return nodata.fill_value(self._data)

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


    def to_cog(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> Path | str:
        """Write this band as a Cloud Optimized GeoTIFF.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            storage_options: Options for the filesystem a URL names.
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
        from geosave_engine.geodata.io import geotiff

        return geotiff.write_cog(
            self.to_raster(),
            path,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def to_gtiff(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[GTiffWriteOptions],
    ) -> Path | str:
        """Write this band as a plain GeoTIFF.

        Reach for `to_cog` unless a consumer needs a striped or otherwise
        non-COG file.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            storage_options: Options for the filesystem a URL names.
            **options: GTiff creation options.

        Returns:
            The written path.

        Raises:
            ValueError: This array names none of the bands it spans, carries
                no `.name` to write a sole band under, or carries a `time`
                dimension rather than a scalar coordinate.
            FileExistsError: `path` exists and `overwrite` is false.
        """
        from geosave_engine.geodata.io.geotiff import write_gtiff

        return write_gtiff(
            self.to_raster(),
            path,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def to_items(
        self,
        path: str | PathLike[str],
        *,
        collection: str | None = None,
        map_scale: float | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> tuple[pystac.Item, ...]:
        """Save this band as COGs and describe them as STAC Items.

        Args:
            path: Name to save the band under, as a local path or fsspec URL
                without a TIFF suffix.
            collection: Name the Items share. None names them after `path`.
            map_scale: Map denominator for TIFF resolution tags.
            overwrite: Replace existing files.
            storage_options: Options for the filesystem a URL names.
            **options: COG creation options.

        Returns:
            One Item per instant, each holding this band as its one asset.

        Raises:
            ValueError: This band is unnamed or timeless.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> [item.id for item in ds["ndvi"].gs.to_items("samples/ndvi")]
            ['ndvi_20250601T103031', 'ndvi_20250611T103031']
        """
        return self.to_raster().gs.to_items(
            path,
            driver="cog",
            collection=collection,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def to_raster(self) -> Dataset:
        """Split this array back into the raster whose variables it stacks.

        The inverse of `GeoRaster.to_array`: each band becomes one variable,
        carrying the attrs that ride on the `band` coordinate. An array
        spanning no band axis becomes a one-variable raster named after it.

        Returns:
            Dataset holding one variable per band. Each band carries its own
            attrs from `StackedAttrs` under the array's own attrs, which every
            band shares; the Dataset root gets the attrs it was stacked from.

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

        stacked = self.attrs.coords[BAND_DIMENSION].get(attrs.StackedAttrs)
        raster = cast("Dataset", self._data.to_dataset(dim=BAND_DIMENSION))
        own = {} if stacked is None else stacked.variable_attrs or {}
        for name in raster.data_vars:
            raster[name].attrs = {**own.get(str(name), {}), **self._data.attrs}
        raster.attrs = {} if stacked is None else dict(stacked.dataset_attrs or {})
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
        fill = attrs.Nodata.from_attrs(self._data.attrs)
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
        from geosave_engine.geodata.utils.color import parse_color

        legend = attrs.Legend.from_attrs(self._data.attrs)
        class_map = None if legend is None else legend.class_map
        if legend is None or class_map is None:
            raise ValueError(
                "band lists no classes, so its values name none to colour; write "
                "a Legend, or compose channels with GeoRaster.to_array"
            )

        colour_of = legend.color_map or {}
        codes = sorted(class_map)
        missing_colour = [code for code in codes if code not in colour_of]
        if missing_colour:
            raise ValueError(
                f"classes {missing_colour} carry no colour; give Legend.color_map an "
                f"entry for every class the band lists"
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
    ) -> hv.Element | hv.NdLayout:
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

        legend = attrs.Legend.from_attrs(self._data.attrs)
        place = self.anchor.locate() if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return plot(
            self._data,
            cmap=cmap,
            clim=clim,
            class_map=legend.class_map if legend else None,
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
