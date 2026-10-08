"""STAC extensions GeoSave writes, one module per schema.

Each module writes its schema's fields onto an asset from a saved raster,
through that schema's PySTAC class, or through GeoSave's own class where
PySTAC ships none.
"""

from . import classification, datacube, eo, projection, raster, zarr
from .geosave import GeosaveExtension
from .zarr import ZarrExtension

# Written onto every asset, in this order: classes sit inside the bands the
# Raster extension stores, so it comes before Classification.
ASSET = (projection, raster, eo, classification, datacube, zarr)

__all__ = [
    "ASSET",
    "GeosaveExtension",
    "ZarrExtension",
    "classification",
    "datacube",
    "eo",
    "projection",
    "raster",
    "zarr",
]
