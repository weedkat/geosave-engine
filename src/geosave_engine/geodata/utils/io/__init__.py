"""Format-specific geodata I/O.

Every format module states `read` and, where the format is writable, `write`.
Reach for them module-qualified — `zarr.read`, `geotiff.write_cog` — so the
format is named once and the verb stays the same across all of them.

Examples:
    >>> from geosave_engine.geodata.utils import io
    >>> raster = io.zarr.read("scene.zarr", mask_and_scale=True)
    >>> io.geotiff.write_cog(raster.gs.to_array(), "scene.tif")
    >>> io.read("scene.zarr")  # by suffix, when the format is not known

The fluent `to_cog`, `to_zarr`, and `to_netcdf` names belong to the `gs`
accessors, not here.
"""

from . import gdal, geojson, geopackage, geoparquet, geotiff, layout, netcdf, zarr
from .dispatch import read
from .gdal import RasterioOpenOptions
from .geojson import GeoJSONOpenOptions
from .geopackage import GeoPackageOpenOptions
from .geoparquet import GeoParquetOpenOptions
from .geotiff import (
    COGWriteOptions,
    GeoTIFFTags,
    GeoTIFFWriteOptions,
    GTiffWriteOptions,
)
from .layout import FlatLayout, Layout, NestedLayout, SAFELayout
from .netcdf import NetCDFOpenOptions, NetCDFWriteOptions
from .zarr import ZarrOpenOptions, ZarrWriteOptions

__all__ = [
    "COGWriteOptions",
    "FlatLayout",
    "GTiffWriteOptions",
    "GeoJSONOpenOptions",
    "GeoPackageOpenOptions",
    "GeoParquetOpenOptions",
    "GeoTIFFTags",
    "GeoTIFFWriteOptions",
    "Layout",
    "NestedLayout",
    "NetCDFOpenOptions",
    "NetCDFWriteOptions",
    "RasterioOpenOptions",
    "SAFELayout",
    "ZarrOpenOptions",
    "ZarrWriteOptions",
    "gdal",
    "geojson",
    "geopackage",
    "geoparquet",
    "geotiff",
    "layout",
    "netcdf",
    "read",
    "zarr",
]
