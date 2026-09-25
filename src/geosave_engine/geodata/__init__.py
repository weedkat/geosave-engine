"""Public geodata surface."""

from typing import TYPE_CHECKING

import xarray as xr

from . import transform
from .core import (
    GeoAnchor,
    GeoArray,
    GeoRaster,
    GeoStack,
    GeoVector,
    raster,
    stack,
)
from .utils import io
from .utils.gdal_env import configure_gdal
from .utils.io import read_raster, read_stack, read_vector
from .utils.io.layout import LAYOUTS, write_tree


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

else:
    DataArray = xr.DataArray
    Dataset = xr.Dataset
    DataTree = xr.DataTree


__all__ = [
    "LAYOUTS",
    "GeoAnchor",
    "GeoArray",
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoRaster",
    "GeoStack",
    "GeoVector",
    "write_tree",
    "configure_gdal",
    "io",
    "raster",
    "read_raster",
    "read_stack",
    "read_vector",
    "stack",
    "transform",
]
