"""StackDataset: PyTorch dataset over GeoStack zarr stores discovered under a root.

SKELETON — being rebuilt for a consistent index-based shape with
StoreDataset (no string key/reindex — see geodata.utils.datasets for the
split-matching helpers that still use one internally). Do not implement
against this yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

LayerName = str


class StackDataset(Dataset):
    """PyTorch dataset over GeoStack zarr stores discovered under root.

    Discovers every `*.zarr` store anywhere under `root` — any depth, so
    anchor stores can be grouped into whatever nested layout makes sense.
    Each store holds one Zarr group per layer (written by `GeoStack.to_zarr`).
    A layer group can be missing from some/all anchors; an anchor store is
    only included if it carries every layer in `required_layers` (None
    means no requirement).

    Args:
        root: Workspace root directory with one subdirectory per anchor.
        required_layers: Layer names to require. None includes every
            anchor folder found, whatever layers it has.
        sel_bands: Layer name to band names to keep; default is all bands.
        dtype_override: Layer name to torch dtype to cast that layer's tensor to.
        key_pattern: Regex to extract each anchor store's split-matching
            key from its name. None strips `.zarr` and uses the rest —
            see `geodata.utils.datasets.extract_key`.
        split: Text file of anchor stems to keep, one per line. None
            keeps every anchor store found under `root`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        required_layers: list[LayerName] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        key_pattern: str | None = None,
        split: str | Path | None = None,
    ) -> None:
        raise NotImplementedError("spec not settled — see module docstring")

    def __len__(self) -> int:
        raise NotImplementedError("spec not settled — see module docstring")

    def render(self, index: int) -> dict[str, Any]:
        """Render one sample. Lazily loads and caches the `GeoStack` at `index`.

        Args:
            index: Row position in this dataset.

        Returns:
            Tensor dict keyed by each layer's raw name, plus `"geobox"`/
            `"geotags"`/the loaded `GeoStack`'s own `context` keys.
        """
        raise NotImplementedError("spec not settled — see module docstring")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("spec not settled — see module docstring")

    def to_row(self, index: int) -> dict[str, Any]:
        """Manifest row for `index` — cheap metadata, no zarr open.

        Args:
            index: Row position in this dataset.

        Returns:
            `{"path": ...}` — the anchor store's path relative to `root`.
        """
        raise NotImplementedError("spec not settled — see module docstring")

    def to_pandas(self) -> pd.DataFrame:
        """Snapshot every sample's `to_row` into one table.

        Raises:
            NotImplementedError: Always — skeleton only.
        """
        raise NotImplementedError("spec not settled — see module docstring")
