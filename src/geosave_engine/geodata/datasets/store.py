"""StoreDataset: PyTorch dataset over a SampleStore's packed samples.

SKELETON — being rebuilt for a consistent index-based shape with
StackDataset; render(index) contract not finalized. No manifest/parquet
methods here — SampleStore already owns to_parquet/fields for the packed
case. Do not implement against this yet.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from geosave_engine.geodata.datastore import SampleStore

LayerName = str


class StoreDataset(Dataset):
    """PyTorch dataset over a SampleStore's packed samples.

    Args:
        store: SampleStore to read from.
        sel_bands: Layer name to band names to keep. Default keeps all
            bands the layer carries.
        dtype_override: Layer name to torch dtype to cast that layer's
            tensor to. Default keeps the stored dtype.
    """

    def __init__(
        self,
        store: SampleStore,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
    ) -> None:
        raise NotImplementedError("spec not settled — see module docstring")

    def __len__(self) -> int:
        raise NotImplementedError("spec not settled — see module docstring")

    def render(self, index: int) -> dict[str, Any]:
        """Sample at index, array fields converted to tensors.

        Args:
            index: Row position in the store.
        """
        raise NotImplementedError("spec not settled — see module docstring")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("spec not settled — see module docstring")
