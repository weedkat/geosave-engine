"""Raster formats: GeoTIFF and COG, Zarr, NetCDF, and Sentinel SAFE products.

`gdal` reads any raster GDAL opens and holds its environment settings.
`geotiff` writes single files, and `cogs` arranges one raster as several of
them.
"""

from . import cogs, gdal, geotiff, netcdf, safe, zarr

__all__ = ["cogs", "gdal", "geotiff", "netcdf", "safe", "zarr"]
