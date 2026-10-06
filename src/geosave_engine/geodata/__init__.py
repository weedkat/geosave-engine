"""Public geodata surface."""

from typing import TYPE_CHECKING

import geopandas as gpd
import xarray as xr

from . import transform
from .core import (
    GeoAnchor,
    GeoArray,
    GeoRaster,
    GeoStack,
    GeoVector,
    GeoRow,
    raster,
    stack,
)
from . import io
from .io.gdal_env import configure_gdal
from .io import read_raster, read_stack, read_vector


if TYPE_CHECKING:

    class DataArray(xr.DataArray):
        """Xarray DataArray carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.DataArray`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoArray

    class Dataset(xr.Dataset):
        """Xarray Dataset carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.Dataset`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoRaster

    class DataTree(xr.DataTree):
        """Xarray DataTree carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.DataTree`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoStack

    class GeoDataFrame(gpd.GeoDataFrame):
        """GeoPandas GeoDataFrame carrying GeoSave's `gs` accessor.

        The runtime value is a `geopandas.GeoDataFrame`; this declaration
        exists only so type checkers can resolve `.gs`.
        """

        gs: GeoVector

else:
    DataArray = xr.DataArray
    Dataset = xr.Dataset
    DataTree = xr.DataTree
    GeoDataFrame = gpd.GeoDataFrame


__all__ = [
    "GeoAnchor",
    "GeoArray",
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoDataFrame",
    "GeoRaster",
    "GeoStack",
    "GeoVector",
    "GeoRow",
    "configure_gdal",
    "io",
    "raster",
    "read_raster",
    "read_stack",
    "read_vector",
    "stack",
    "transform",
]
