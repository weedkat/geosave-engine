"""StackDataset: PyTorch dataset over GeoStack zarr stores discovered under a root.

See docs/concept/model.md for the settled design this implements.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from geosave_engine.geodata.spatial.stack import GeoStack

LayerName = str


class StackDataset(Dataset):
    """PyTorch dataset over GeoStack zarr stores discovered under root.

    Discovers every `*.zarr` store anywhere under `root` — any depth, so
    anchor stores can be grouped into whatever nested layout makes sense.
    Each store holds one Zarr group per layer (written by `GeoStack.to_zarr`).
    A layer group can be missing from some/all anchors; an anchor store is
    only included if it carries every layer in `required_layers` (None
    means no requirement) — anchors that pass keep every layer they carry,
    not just the required ones.

    Every discovered anchor is opened once here with `load_data=False`
    (header-only — geobox/datetime read from attrs, no pixels touched) and
    cached, so building the index over a large `root` is cheap; `render`
    reuses the cached `GeoStack` and only then materializes pixels.

    Args:
        root: Workspace root directory with one subdirectory per anchor.
        required_layers: Layer names to require. None includes every
            anchor folder found, whatever layers it has.
        sel_bands: Layer name to band names to keep; default is all bands.
        dtype_override: Layer name to torch dtype to cast that layer's tensor to.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        required_layers: list[LayerName] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
    ) -> None:
        self.root = Path(root)
        self.sel_bands = sel_bands
        self.dtype_override = dtype_override

        required = set(required_layers) if required_layers else set()
        samples: list[tuple[Path, GeoStack]] = []
        for path in sorted(self.root.rglob("*.zarr")):
            stack = GeoStack.from_zarr(path, load_data=False)
            if required <= set(stack.tiles):
                samples.append((path, stack))
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def render(self, index: int) -> dict[str, Any]:
        """Render one sample. Lazily loads and caches the `GeoStack` at `index`.

        Args:
            index: Row position in this dataset.

        Returns:
            Tensor dict keyed by each layer's raw name, plus `"geobox"`/
            `"geotags"`/the loaded `GeoStack`'s own `context` keys.
        """
        _, stack = self._samples[index]
        return stack.to_tensor(self.sel_bands, self.dtype_override)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.render(index)

    def to_row(self, index: int) -> dict[str, Any]:
        """Manifest row for `index` — cheap metadata, no zarr open.

        Args:
            index: Row position in this dataset.

        Returns:
            `{"path": ...}` — the anchor store's path relative to `root`.
        """
        path, _ = self._samples[index]
        return {"path": str(path.relative_to(self.root))}

    def to_pandas(self) -> pd.DataFrame:
        """Snapshot every sample's `to_row` into one table.

        Returns:
            DataFrame, one row per sample, in index order.
        """
        return pd.DataFrame([self.to_row(i) for i in range(len(self))])
