"""PyTorch dataset classes over complete, standalone samples.

Two kinds, both index-based (no string key/join layer): `StackDataset`
reads raw `GeoStack` zarr directly off disk; `StoreDataset` reads a packed
`SampleStore`. A dataset here always renders a complete sample on its
own — combine sources upstream (e.g. into one `SampleStore`) instead of
downstream. Custom formats not covered by either: write your own
`torch.utils.data.Dataset`.

SKELETON — both classes are being rebuilt from scratch for a consistent
shape, see their own module docstrings.

ML-input only — samples are keyed and materialized whole for training/inference.
For windowed access to a big raster (serving/viewing), see geodata.datastore instead.
"""
from .stack import StackDataset
from .store import StoreDataset
from .samplers import stack_samples

__all__ = [
    "StackDataset",
    "StoreDataset",
    "stack_samples",
]
