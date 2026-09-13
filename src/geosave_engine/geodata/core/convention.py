"""The dimension and coordinate names a GeoSave xarray object carries.

odc-geo names the spatial dimensions from the CRS, and every name below sits
beside them: `band` where variables or channels stack, `time` where
observations repeat, `spatial_ref` where the CRS is written.

Examples:
    A DataArray holds one band's pixels. `band` and `time` lead, the spatial
    pair trails, and `spatial_ref` rides along carrying the CRS:

    >>> array(pixels, geobox, time=labels, band=["red", "green", "blue"])
    <xarray.DataArray (time: 2, band: 3, y: 512, x: 512)>
    Coordinates:
      * time         (time) datetime64[ns] 2025-06-01 2025-06-11
      * band         (band) <U5 'red' 'green' 'blue'
      * y            (y) float64 5.005e+06 5.005e+06 ... 5e+06
      * x            (x) float64 3e+05 3e+05 ... 3.051e+05
        spatial_ref  int32 32633

    A Dataset holds several variables on one grid. They need not span the same
    axes — `red` repeats over `time`, `dem` does not:

    >>> raster({"red": cube, "dem": (flat, ())}, geobox, time=labels)
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

    >>> array(pixels)
    <xarray.DataArray (y: 512, x: 512)>
    Dimensions without coordinates: y, x

A geographic CRS names the spatial pair `latitude` and `longitude` instead,
which odc supplies and `CFCoordinate.from_geobox` describes.
"""

from __future__ import annotations

# Dimensions a raster spans when odc resolves no grid, odc's own first choice.
NOT_GEOREFERENCED_DIMENSIONS = ("y", "x")

# Scalar coordinate holding the CRS, which CF reaches through `grid_mapping`.
CRS_COORDINATE = "spatial_ref"

# Axis the bands occupy: a file's bands, stacked variables, or display channels.
BAND_DIMENSION = "band"

# Axis a raster spans when it carries observations over time.
TIME_COORDINATE = "time"

# Coordinate holding the two edges of the bucket each `time` label stands for.
TIME_BOUNDS_COORDINATE = "time_bnds"

# Axis a bounds coordinate spans, one position per cell edge.
BOUNDS_DIMENSION = "bnds"
