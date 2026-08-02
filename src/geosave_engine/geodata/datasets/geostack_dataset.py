from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch

from geosave_engine.geodata.datasets.base_dataset import BaseDataset, extract_key, filter_by_split
from geosave_engine.geodata.tile import GEOSTACK_SUFFIX, GeoStack, GeoTile

log = logging.getLogger(__name__)

LayerName = str


class GeoStackDataset(BaseDataset):
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
        >>> ds = GeoStackDataset("data/train")
        >>> loader = DataLoader(ds, batch_size=4, collate_fn=stack_samples)
        >>> batch = next(iter(loader))  # {"sentinel_2_l1c": Tensor[4,C,H,W], "dynamicworld": Tensor[4,1,H,W]}

        >>> # predict split with no labels: only require the layers that exist
        >>> predict_ds = GeoStackDataset("data/predict", required_layers=["sentinel_2_l1c", "cloud_mask"])
    """

    def __init__(
        self,
        root: str | Path,
        *,
        required_layers: list[LayerName] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        context_fn: Callable[[dict[LayerName, GeoTile]], dict[str, torch.Tensor]] | None = None,
        key_pattern: str | None = None,
        split: str | Path | None = None,
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
            key_pattern: Regex to extract the sample key from each anchor
                folder's name. None strips `.geostack` and uses the rest —
                see `extract_key`.
            split: Text file of anchor stems to keep, one per line. None
                keeps every anchor folder found under `root`.
        """
        self.split = split
        self.root = Path(root)
        self.required_layers = required_layers
        self.sel_bands = sel_bands
        self.dtype_override = dtype_override
        self.context_fn = context_fn
        self.key_pattern = key_pattern
        self._cache: dict[str, GeoStack] = {}

        anchor_dirs = sorted(self.root.rglob(f"*{GEOSTACK_SUFFIX}"))
        paths = {extract_key(anchor_dir.name, key_pattern): anchor_dir for anchor_dir in anchor_dirs}
        paths = filter_by_split(paths, split)

        if required_layers is not None:
            # cheap: just the folder's own file names, no GeoStack.load yet
            paths = {
                stem: anchor_dir
                for stem, anchor_dir in paths.items()
                if set(required_layers).issubset({p.stem for p in anchor_dir.glob("*.zarr")})
            }

        self.paths = paths
        self.reindex(paths)

    def render(self, key: str) -> dict[str, Any]:
        """Render one sample. Lazily loads and caches the `GeoStack` for `key`.

        Args:
            key: Anchor folder stem — one of `self.keys`.

        Returns:
            Tensor dict keyed by each layer's raw name, plus ``"anchor"``
            (and whatever `self.context_fn` returns, if set).
        """
        if key not in self._cache:
            self._cache[key] = GeoStack.load(self.paths[key], required_layers=self.required_layers, load_data=False)
        return self._cache[key].to_tensor(self.sel_bands, self.dtype_override, self.context_fn)

    def to_row(self, key: str) -> dict[str, Any]:
        """Manifest row for `key`.

        Args:
            key: Anchor folder stem — one of `self.keys`.

        Returns:
            `{"path": ...}` — `self.paths[key]` relative to `self.root`.
        """
        return {"path": str(self.paths[key].relative_to(self.root))}
