"""Moving a raster's pixels onto another grid.

`odc.stac.load` warps at load time, given a geobox. These are the same
operation for a raster already in memory, which has no load to hook into.

Examples:
    A delivered DEM has to reach the imagery's grid before it can join a
    stack, and a quicklook wants a CRS rather than an exact grid::

        dem = reproject_match(srtm, scene, resampling="bilinear")
        webmap = reproject(scene, "EPSG:3857")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs

if TYPE_CHECKING:
    from odc.geo import SomeCRS, SomeResolution

# Kernel names odc forwards to rasterio.warp.Resampling.
type Resampling = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "gauss",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
    "rms",
]

# Kernels that carry a stored value across rather than blending several.
_VALUE_PRESERVING = frozenset({"nearest", "mode"})


def reproject_match[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    match: GeoBox | xr.DataArray | xr.Dataset | xr.DataTree,
    *,
    resampling: Resampling = "nearest",
) -> T:
    """Warp pixels onto a grid that already exists.

    Every group of a DataTree lands on the same grid, so a stack keeps the
    single grid it requires.

    Args:
        data: DataArray, Dataset, or DataTree carrying a locatable grid.
        match: Grid to land on, or any xarray object carrying one. Its CRS,
            resolution, and extent are all adopted, so no resolution is taken.
        resampling: GDAL resampling kernel.

    Returns:
        New object of the same kind on the matched grid, carrying its own
        `spatial_ref` and the CF semantics its axes earn.

    Raises:
        TypeError: `data` is not an xarray object this warps.
        ValueError: `data` carries no locatable grid, a variable does not span
            it, or `resampling` would blend a variable whose values are class
            codes.

    Examples:
        The target grid comes from whatever already defines it — another
        raster, or the anchor the workspace loaded against:

        >>> dem = reproject_match(srtm, scene, resampling="bilinear")
        >>> dem.gs.geobox == scene.gs.geobox
        True

        Class codes must not be averaged, so a blending kernel refuses:

        >>> reproject_match(landcover, scene, resampling="bilinear")
        Traceback (most recent call last):
        ValueError: ['landcover'] carry a class map, so 'bilinear' would blend ...
    """
    if not isinstance(data, xr.DataArray | xr.Dataset | xr.DataTree):
        raise TypeError(
            f"warping takes a DataArray, Dataset, or DataTree, got "
            f"{type(data).__name__}"
        )

    geobox = match if isinstance(match, GeoBox) else match.gs.geobox

    if resampling not in _VALUE_PRESERVING:
        flagged = attrs.flag_variables(data)
        if flagged:
            raise ValueError(
                f"{list(flagged)} carry a class map, so {resampling!r} would blend their "
                f"codes into values naming no class; warp them with 'nearest' or "
                f"'mode', or drop the Legend first"
            )

    if isinstance(data, xr.DataArray):
        return cast("T", data.odc.reproject(geobox, resampling=resampling))

    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import stack

        return cast(
            "T",
            stack(
                {
                    name: reproject_match(raster, geobox, resampling=resampling)
                    for name, raster in data.gs.rasters.items()
                }
            ),
        )

    # odc's Dataset path keeps the old spatial_ref attrs, so it reports the source.
    raster = cast("xr.Dataset", data)
    grid_dims = set(raster.gs.grid_dims)
    warped: dict[str, xr.DataArray] = {}
    for name, array in raster.data_vars.items():
        if not grid_dims <= set(array.dims):
            raise ValueError(
                f"{str(name)!r} spans {list(array.dims)}, not the grid "
                f"{sorted(grid_dims)}, so it names no pixels to warp; drop it "
                f"or place it on the grid first"
            )
        warped[str(name)] = array.odc.reproject(geobox, resampling=resampling)
    return cast("T", xr.Dataset(warped, attrs=dict(raster.attrs)).gs.write_crs())


def reproject[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    crs: SomeCRS,
    *,
    resampling: Resampling = "nearest",
    resolution: SomeResolution | None = None,
) -> T:
    """Warp pixels into another CRS, deriving the grid to land on.

    Reach for `reproject_match` where the target grid already exists; this is
    for when only the CRS is decided and the grid is sized from the source.

    Args:
        data: DataArray, Dataset, or DataTree carrying a locatable grid.
        crs: Coordinate reference system to land in.
        resampling: GDAL resampling kernel.
        resolution: Output pixel size. None keeps the source's ground sampling
            as closely as the new CRS allows.

    Returns:
        New object of the same kind in `crs`, on a grid covering the source.

    Raises:
        TypeError: `data` is not an xarray object this warps.
        ValueError: `data` carries no locatable grid, a variable does not span
            it, or `resampling` would blend a variable whose values are class
            codes.

    Examples:
        >>> reproject(scene, "EPSG:3857").gs.crs.epsg
        3857
        >>> reproject(scene, "EPSG:4326", resolution=0.001).gs.resolution.x
        0.001
    """
    # odc spells "size it for me" as "auto".
    scale = "auto" if resolution is None else resolution
    return reproject_match(
        data, data.gs.geobox.to_crs(crs, resolution=scale), resampling=resampling
    )
