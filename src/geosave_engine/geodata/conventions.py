"""Conventions for the dimension and coordinate names a GeoSave xarray object carries.

Every raster spans `y` and `x`, and every name below sits beside them: `band` where variables or channels stack, `time` where
observations repeat, `spatial_ref` where the CRS is written.

Examples:
    A DataArray holds one band's pixels. `band` and `time` lead, the spatial
    pair trails, and `spatial_ref` rides along carrying the CRS:

    >>> array(pixels, geobox, dims=("time", "band", "y", "x"),
    ...       coords={"time": labels, "band": ["red", "green", "blue"]})
    <xarray.DataArray (time: 2, band: 3, y: 512, x: 512)>
    Coordinates:
      * time         (time) datetime64[ns] 2025-06-01 2025-06-11
      * band         (band) <U5 'red' 'green' 'blue'
      * y            (y) float64 5.005e+06 5.005e+06 ... 5e+06
      * x            (x) float64 3e+05 3e+05 ... 3.051e+05
        spatial_ref  int32 32633

    A Dataset holds several variables on one grid. They need not span the same
    axes — `red` repeats over `time`, `dem` does not:

    >>> raster({"red": (("time", "y", "x"), cube),
    ...         "dem": (("y", "x"), flat)}, geobox, coords={"time": labels})
    <xarray.Dataset>
    Dimensions:      (y: 512, x: 512, time: 2)
    Coordinates:
      * y            (y) float64 5.005e+06 5.005e+06 ... 5e+06
      * x            (x) float64 3e+05 3e+05 ... 3.051e+05
      * time         (time) datetime64[ns] 2025-06-01 2025-06-11
        spatial_ref  int32 32633
    Data variables:
        red          (time, y, x) uint16 ...
        dem          (y, x) float32 ...

    A DataTree holds several rasters on one shared grid. The root carries the
    coordinates every group shares, so a group repeats only `spatial_ref` and
    xarray refuses one that does not align:

    >>> stack({"sentinel-2-l2a": optical, "dem": dem})
    <xarray.DataTree>
    Group: /
    │   Dimensions:      (y: 512, x: 512, time: 2)
    │   Coordinates:
    │     * y            (y) float64 5.005e+06 5.005e+06 ... 5e+06
    │     * x            (x) float64 3e+05 3e+05 ... 3.051e+05
    │     * time         (time) datetime64[ns] 2025-06-01 2025-06-11
    │       spatial_ref  int32 32633
    ├── Group: /sentinel-2-l2a
    │       Dimensions:      (time: 2, y: 512, x: 512)
    │       Coordinates:
    │           spatial_ref  int32 32633
    │       Data variables:
    │           red          (time, y, x) uint16 ...
    └── Group: /dem
            Dimensions:      (y: 512, x: 512)
            Coordinates:
                spatial_ref  int32 32633
            Data variables:
                dem          (y, x) float32 ...

    Without a geobox odc resolves no grid, so the pixels keep their shape and
    claim no ground position. The spatial pair is named but carries no labels:

    >>> array(pixels, dims=("y", "x"))
    <xarray.DataArray (y: 512, x: 512)>
    Dimensions without coordinates: y, x

Every raster spans `y` and `x`, whatever its CRS. Geographic coordinates carry
CF standard names `latitude` and `longitude` and degree units; projected
coordinates carry projection standard names and linear units. CF reads those
attrs, not the dimension names. Readers rename a foreign grid on the way in
with `to_yx`, and the `gs` accessors refuse one with `require_yx`.
"""

from __future__ import annotations

from typing import cast

import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr
from odc.geo.xr import spatial_dims
from rioxarray.exceptions import MissingSpatialDimensionError

# Spatial dimensions of every raster, georeferenced or not.
SPATIAL_DIMENSIONS = ("y", "x")

# Scalar coordinate holding the CRS, which CF reaches through `grid_mapping`.
CRS_COORDINATE = "spatial_ref"

# Axis the bands occupy: a file's bands, stacked variables, or display channels.
BAND_DIMENSION = "band"

# Axis a raster spans when it carries observations over time.
TIME_COORDINATE = "time"


def _spatial_pair(data: xr.DataArray | xr.Dataset) -> tuple[str, str] | None:
    """Find the spatial pair by a conventional name, then by its CF attrs."""
    dims = spatial_dims(data)  # y, x | latitude, longitude | lat, lon
    if dims is None:
        try:
            # Any name, where the coordinates carry CF `axis` or `standard_name`.
            dims = (str(data.rio.y_dim), str(data.rio.x_dim))
        except MissingSpatialDimensionError:
            return None
    return dims


def to_yx[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Rename a foreign spatial pair to `y` and `x`.

    The pair is found by name (`latitude`/`longitude`, `lat`/`lon`) or, under
    any other name, by the CF `axis` or `standard_name` its coordinates carry.

    Args:
        data: Band, raster, or stack as a loader named its grid.

    Returns:
        Renamed view sharing its pixels, lazily. A band or raster already on
        `y` and `x`, or carrying no grid, comes back as itself.

    Examples:
        >>> to_yx(xr.open_dataset("era5.nc")).t2m.dims
        ('time', 'y', 'x')
    """
    if isinstance(data, xr.DataTree):
        return cast("T", data.map_over_datasets(to_yx))

    dims = _spatial_pair(data)
    if dims in (None, SPATIAL_DIMENSIONS):
        return data
    return cast("T", data.rename(dict(zip(dims, SPATIAL_DIMENSIONS, strict=True))))


def require_yx(data: xr.DataArray | xr.Dataset | xr.DataTree) -> None:
    """Refuse a grid spanning anything but `y` and `x`.

    Args:
        data: Band, raster, or stack about to be read through `gs`.

    Raises:
        ValueError: Its spatial dimensions, or a group's, carry another pair.
    """
    if isinstance(data, xr.DataTree):
        for node in data.subtree:
            require_yx(node.dataset)
        return

    dims = _spatial_pair(data)
    if dims not in (None, SPATIAL_DIMENSIONS):
        fix = dict(zip(dims, SPATIAL_DIMENSIONS, strict=True))
        raise ValueError(
            f"the grid spans {dims}, but gs reads {SPATIAL_DIMENSIONS} for every "
            f"CRS; rename it first with .rename({fix})"
        )
