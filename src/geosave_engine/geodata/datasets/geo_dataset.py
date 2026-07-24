from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from geosave_engine.geodata.tile import GEOSTACK_SUFFIX, GeoStack, GeoTile

log = logging.getLogger(__name__)

LayerName = str


class GeoDataset(Dataset):
    """Georeferenced PyTorch dataset over ``GeoStack``s under a workspace root.

    Discovers every ``*.geostack`` folder anywhere under ``root`` — any
    depth, so anchor folders can be grouped into whatever nested layout
    makes sense (e.g. mirroring raw data provenance:
    ``root/train/Experts/EH/1/<anchor>.geostack/``), not just flat directly
    under ``root``::

        data/train/
        ├── 13.000000_52.000000_20240115T000000_20240115T235959.999999_10m.geostack/
        │   ├── sentinel_2_l1c.zarr
        │   ├── cloud_mask.zarr
        │   └── dynamicworld.zarr
        └── Experts/EH/1/13.320000_52.000000_20240115T000000_20240115T235959.999999_10m.geostack/
            ├── sentinel_2_l1c.zarr
            ├── cloud_mask.zarr
            └── dynamicworld.zarr

    Two anchor folders here — ``len(ds) == 2``. A layer store can be missing
    from some/all anchors (e.g. no ``dynamicworld`` on a label-less predict
    split); an anchor folder is only included if it carries every layer in
    ``required_layers`` (None means no requirement — every anchor folder found
    is included, whatever layers it happens to have).

    Examples:
        >>> ds = GeoDataset("data/train")
        >>> loader = DataLoader(ds, batch_size=4, collate_fn=stack_samples)
        >>> batch = next(iter(loader))  # {"sentinel_2_l1c": Tensor[4,C,H,W], "dynamicworld": Tensor[4,1,H,W]}

        >>> # predict split with no labels: only require the layers that exist
        >>> predict_ds = GeoDataset("data/predict", required_layers=["sentinel_2_l1c", "cloud_mask"])
    """

    def __init__(
        self,
        root: str | Path,
        *,
        required_layers: list[LayerName] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        context_fn: Callable[[dict[LayerName, GeoTile]], dict[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            root: Workspace root directory with one subdirectory per anchor.
            required_layers: Layer names to require. None includes every
                anchor folder found, whatever layers it has.
            sel_bands: Layer name to band names to keep; default is all bands.
            dtype_override: Layer name to torch dtype to cast that layer's tensor to.
                Only needed to deviate from the tensor's saved dtype (e.g. cast a
                uint8 mask layer to bool).
            context_fn: Optional, forwarded to `GeoStack.to_tensor` on every
                `__getitem__` call — see there for what it receives/returns.
        """
        self.root = Path(root)
        self.sel_bands = sel_bands
        self.dtype_override = dtype_override
        self.context_fn = context_fn

        anchor_dirs = sorted(self.root.rglob(f"*{GEOSTACK_SUFFIX}"))
        samples: list[GeoStack] = []
        for anchor_dir in anchor_dirs:
            available = {p.stem for p in anchor_dir.glob("*.zarr")}
            if required_layers is not None and not set(required_layers).issubset(available):
                continue
            # lazy load each geostack, memory friendly
            samples.append(GeoStack.load(anchor_dir, required_layers=required_layers))
        self.samples = samples

        if not self.samples:
            log.warning("Empty dataset: no anchor folders found under %s", self.root)
        else:
            log.info("GeoDataset layers: %s", self.layers)

    @property
    def layers(self) -> list[LayerName]:
        """Layer names carried by this dataset's samples, in discovery order."""
        if not self.samples:
            return []
        return list(self.samples[0].tiles)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Render one sample.

        Args:
            index: Row index into the sample index.

        Returns:
            Tensor dict keyed by each layer's raw name, plus ``"anchor"``
            (and whatever `self.context_fn` returns, if set).
        """
        return self.samples[index].to_tensor(self.sel_bands, self.dtype_override, self.context_fn)
