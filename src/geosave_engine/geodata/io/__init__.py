"""One module per file format, each reading and writing that format alone.

Every module has a `read`; a writable format also has a `write`, except GeoTIFF
(`write_cog`, `write_gtiff`) and `cogs`, which arranges one raster as several
GeoTIFFs that `read_raster` reads back. Reach for them module-qualified:

Examples:
    >>> from geosave_engine.geodata import io
    >>> raster = io.zarr.read("scene.zarr", mask_and_scale=True)
    >>> io.geotiff.write_cog(raster, "scene.tif")
    >>> io.read_raster("scene.zarr")  # by suffix, when the format is not known

The fluent `to_cog`, `to_zarr`, and `to_netcdf` names belong to the `gs`
accessors, not here.
"""

from . import (
    cogs,
    gdal,
    geojson,
    geopackage,
    geoparquet,
    geotiff,
    netcdf,
    safe,
    zarr,
)
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
from .netcdf import NetCDFOpenOptions, NetCDFWriteOptions
from .zarr import ZarrOpenOptions, ZarrWriteOptions
from .gdal_env import configure_gdal
from .readers import read_raster, read_stack, read_vector

__all__ = [
    "COGWriteOptions",
    "GTiffWriteOptions",
    "GeoJSONOpenOptions",
    "GeoPackageOpenOptions",
    "GeoParquetOpenOptions",
    "GeoTIFFTags",
    "GeoTIFFWriteOptions",
    "NetCDFOpenOptions",
    "NetCDFWriteOptions",
    "RasterioOpenOptions",
    "ZarrOpenOptions",
    "ZarrWriteOptions",
    "configure_gdal",
    "cogs",
    "gdal",
    "geojson",
    "geopackage",
    "geoparquet",
    "geotiff",
    "netcdf",
    "read_raster",
    "read_stack",
    "read_vector",
    "safe",
    "zarr",
]
