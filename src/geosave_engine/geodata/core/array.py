"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike
    from pathlib import Path

    import holoviews as hv
    from odc.geo import CRS
    from typing_extensions import Unpack

    from geosave_engine.geodata.attrs import AttrsHeader
    from geosave_engine.geodata.utils.datetime import DateRange
    from geosave_engine.geodata.utils.io.geotiff import (
        COGWriteOptions,
        GTiffWriteOptions,
    )

    from geosave_engine.geodata import DataArray

    from .anchor import GeoAnchor

# Spatial dims an unplaced array carries, which no grid names for it.
UNPLACED_DIMENSIONS = ("y", "x")

# Coordinate odc writes the CRS onto, which a leading axis may not shadow.
_CRS_COORDINATE = "spatial_ref"

# Axis a bucketed raster spans.
_TIME_COORDINATE = "time"

# Axis the bands occupy: a file's own bands, stacked variables, or display channels.
BAND_DIMENSION = "band"


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
    along the axes `coords` declares ahead of them. Build several bands at
    once with `raster`, which returns them as one Dataset.

    Args:
        pixels: Values, ending in the two spatial axes.
        geobox: Grid placing the trailing two axes. None leaves the band
            unplaced, so it holds pixels without claiming ground position.
        nodata: Value standing for absent pixels, declared as
            `Packing.fill_value`. None declares none.
        **coords: Leading axis name mapped to its labels, in array order. None
            labels an axis carrying none. Pass a name Python reserves as
            `**{"class": labels}`.

    Returns:
        Placed DataArray when `geobox` is given, its spatial coordinates
        naming what they measure, otherwise an unplaced DataArray whose
        spatial dims are named `y` and `x`.

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

    spatial_dims = UNPLACED_DIMENSIONS if geobox is None else geobox.dimensions
    grid_coords = (*spatial_dims, _CRS_COORDINATE)
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
        built = attrs.rebase(built, attrs.Packing(_FillValue=nodata))

    if geobox is not None:
        # odc places the axes; GeoSave states what they measure.
        for name, semantics in attrs.CFCoordinate.from_geobox(geobox).items():
            built = attrs.rebase(built, semantics, target=name)

    return cast("DataArray", built)


@xr.register_dataarray_accessor("gs")
class GeoArray:
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
        self._data: DataArray = cast("DataArray", data)

    @property
    def geobox(self) -> GeoBox:
        """Read the pixel grid, which places this band on the ground.

        Returns:
            Grid including CRS, transform, bounds, and shape.

        Raises:
            ValueError: The band declares no CRS, carries no spatial dims, or
                describes its grid by ground control points rather than a
                transform.
        """
        grid = self._data.odc.geobox
        if grid is None:
            raise ValueError(
                "band carries no locatable grid; it declares no CRS or spatial dims"
            )
        if not isinstance(grid, GeoBox):
            raise ValueError(
                f"band grid is a {type(grid).__name__}; expected a regular GeoBox"
            )
        if grid.crs is None:
            raise ValueError(
                "band grid declares no CRS, so it is indexed in pixels rather than "
                "placed on the ground; assign one with odc.geo.xr.assign_crs before "
                "asking where it is"
            )
        return grid

    @property
    def crs(self) -> CRS:
        """Read the coordinate reference system.

        Returns:
            CRS declared by the grid mapping coordinate.

        Raises:
            ValueError: The band declares no CRS.
        """
        return cast("CRS", self.geobox.crs)  # geobox already refused a CRS-less grid

    @property
    def grid_dims(self) -> tuple[str, str]:
        """Name the two dimensions the grid spans.

        Returns:
            `("y", "x")` for a projected CRS, `("latitude", "longitude")` for
            a geographic one, and `("y", "x")` for an unplaced band, whose
            pixels span those axes without claiming ground position.
        """
        grid = self._data.odc.geobox
        return UNPLACED_DIMENSIONS if grid is None else grid.dimensions

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
    def attrs(self) -> AttrsHeader:
        """Read the typed attrs this band carries.

        Returns:
            Detached header whose `root` holds the band's own models, reread on
            every access because xarray attrs are mutable in place.
        """
        return attrs.read(self._data)

    @property
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage.

        Returns:
            First and last covered instant, or None for timeless data.
        """
        if _TIME_COORDINATE not in self._data.coords:
            return None
        spec = self.attrs.variables[_TIME_COORDINATE].get(attrs.TimeSpec)
        labels = self._data.coords[_TIME_COORDINATE].values
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

    def colorize(self) -> DataArray:
        """Bake this band's declared class colours into display channels.

        Reads `Legend`, colouring each pixel by the class its code names. The
        result holds colour rather than codes, so it draws without a legend;
        prefer `plot` where a named colorbar is wanted.

        Returns:
            Placed array shaped `(*axes, band, y, x)` valued in
            `[0, 1]`, whose `band` coordinate is `("red", "green", "blue")`.
            A pixel no class names is absent on every channel.

        Raises:
            ValueError: The band declares no `Legend.class_map`, or names a
                class carrying no colour.

        Examples:
            >>> ds["landcover"].gs.colorize().sizes["band"]
            3
        """
        from geosave_engine.utils.colorize import parse_color

        legend = self.attrs.root.get(attrs.Legend)
        if not isinstance(legend, attrs.Legend) or legend.class_map is None:
            raise ValueError(
                "band declares no Legend.class_map, so its values name no classes "
                "to colour; declare one, or compose channels with GeoRaster.to_array"
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
        declaring `Legend.class_map` draws through that palette with its classes
        named on the colorbar, while `viz.continuous` draws the bare codes.

        Args:
            cmap: Colormap. Refused for a band declaring a class map, whose
                palette is `Legend.color_map`.
            clim: Colour limits. Refused for a band declaring a class map,
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
                is given for a band declaring a class map, or its attrs
                contradict each other.

        Examples:
            >>> ds["ndvi"].gs.plot(cmap="RdYlGn")
            >>> ds["ndvi"].gs.plot(cols=6)  # one panel per bucket
        """
        from geosave_engine.geodata.viz import plot

        legend = self.attrs.root.get(attrs.Legend)
        declared = legend if isinstance(legend, attrs.Legend) else None
        return plot(
            self._data,
            cmap=cmap,
            clim=clim,
            class_map=declared.class_map if declared else None,
            color_map=declared.color_map if declared else None,
            cols=cols,
            title=title,
            xlabel=xlabel,
        )

    def to_cog(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
        **options: Unpack[COGWriteOptions],
    ) -> Path:
        """Write this array as one Cloud Optimized GeoTIFF.

        Band labels become GDAL band descriptions and a scalar `time`
        coordinate becomes ``TIFFTAG_DATETIME``, so the file names what it
        holds. State tags with `rebase(array, GeoTIFFTags(...))` first.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            overwrite: Replace an existing file when true.
            **options: COG creation options.

        Returns:
            The written path.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong, or the array spans a `time`
                dimension rather than one instant.

        Examples:
            >>> ds.gs.to_array(["ndvi"]).gs.to_cog("ndvi.tif")
            PosixPath('ndvi.tif')
        """
        from geosave_engine.geodata.utils.io import geotiff

        return geotiff.write_cog(self._data, path, overwrite=overwrite, **options)

    def to_gtiff(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
        **options: Unpack[GTiffWriteOptions],
    ) -> Path:
        """Write this array as one plain GeoTIFF.

        Reach for `to_cog` unless a consumer needs a striped or otherwise
        non-COG file. Coordinates map to tags exactly as `to_cog` maps them.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            overwrite: Replace an existing file when true.
            **options: GTiff creation options.

        Returns:
            The written path.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong, or the array spans a `time`
                dimension rather than one instant.

        Examples:
            >>> ds.gs.to_array(["ndvi"]).gs.to_gtiff("ndvi.tif", tiled=True)
            PosixPath('ndvi.tif')
        """
        from geosave_engine.geodata.utils.io import geotiff

        return geotiff.write_gtiff(self._data, path, overwrite=overwrite, **options)
