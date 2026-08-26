"""PyTorch dataset classes over ingested surfaces, one sample per window.

`RasterDataset` reads standalone raster files, `StackDataset` this library's
own multi-layer zarr stores, `StoreDataset` a packed `LitDataStore`.
"""
from .litdata import StoreDataset
from .raster import RasterDataset
from .samplers import stack_samples
from .stack import StackDataset

__all__ = [
    "RasterDataset",
    "StackDataset",
    "StoreDataset",
    "stack_samples",
]
