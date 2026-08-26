from collections.abc import Callable

import xarray as xr

from .archives import cleanup_zip, extract_zip
from .gdal import configure_gdal
from .netcdf import from_netcdf, to_netcdf
from .options import GeotiffOptions, NetcdfFormat, NetcdfOptions, ZarrOptions
from .rasterio import BAND_NAME_ATTR, from_rasterio, to_geotiff
from .vector import (
    SIDECAR_SUFFIX,
    VECTOR_SUFFIXES,
    from_vector,
    read_sidecar,
    sidecar_path,
    to_geoparquet,
    write_sidecar,
)
from .zarr import from_zarr, to_zarr

# Suffixes whose reader takes a `group=`; every other format holds exactly one raster per file.
GROUPED_SUFFIXES = frozenset({".zarr", ".nc"})

# GeoRaster.open dispatches on this. Supporting a new format means one reader module + one entry here.
READERS: dict[str, Callable[..., xr.DataArray]] = {
    ".zarr": from_zarr,
    ".nc": from_netcdf,
    ".tif": from_rasterio,
    ".tiff": from_rasterio,
    ".cog": from_rasterio,
    ".png": from_rasterio,
    ".jpg": from_rasterio,
    ".jpeg": from_rasterio,
}

__all__ = [
    "BAND_NAME_ATTR",
    "GROUPED_SUFFIXES",
    "READERS",
    "SIDECAR_SUFFIX",
    "VECTOR_SUFFIXES",
    "read_sidecar",
    "sidecar_path",
    "write_sidecar",
    "GeotiffOptions",
    "NetcdfFormat",
    "NetcdfOptions",
    "ZarrOptions",
    "cleanup_zip",
    "configure_gdal",
    "extract_zip",
    "from_rasterio",
    "from_netcdf",
    "from_zarr",
    "from_vector",
    "to_geotiff",
    "to_geoparquet",
    "to_netcdf",
    "to_zarr",
]
