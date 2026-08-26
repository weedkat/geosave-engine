"""StackDataset: PyTorch dataset over GeoStack zarr stores. See StackDataset for details."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from torch.utils.data import Dataset

from geosave_engine.geodata.spatial import (
    ContextFn,
    GeoTileStack,
    GeoStack,
    LayerName,
    TensorSample,
    TimeWindow,
)

if TYPE_CHECKING:
    from geosave_engine.geodata.extensions import TilerMode

# A zarr store keeps one directory per group, so its layers are readable without opening it.
STORE_SUFFIX = ".zarr"


class StackDataset(Dataset[TensorSample]):
    """PyTorch dataset over ingested GeoStack stores, one sample per tile.

    An ingested store holds a whole anchor's surface, so leave `tile_size_px`
    unset only when every store is already sample-sized. A store this
    configuration can't cut is warned about and skipped.

    Args:
        root: Directory holding `.zarr` stores, searched recursively.
        sel_bands: Layer name to band names to keep, in that order. A layer
            absent here keeps every band.
        dtype_override: Layer name to torch dtype to cast to. A layer absent
            here keeps its stored dtype.
        layers: Layer names to keep from every store. None keeps all.
        required_layers: A store missing any of these is left out of the
            index. Read from the store's own group directories, so no store
            is opened to decide.
        tile_size_px: Window side length in pixels. None uses the shorter
            axis, so each store yields one whole-surface sample.
        stride_px: Distance between window origins. None = `tile_size_px`.
        overlap: Forwarded to the tiler. Wins over `stride_px`.
        mode: How a trailing window's overhang is filled — "reflect",
            "edge", or "constant", which needs a declared nodata.
        vector: True gives each sample the reference layer's features,
            filtered to the window.
        time: `(length, stride)` in reference-layer steps, or a bare length.
            Windows are cut on the reference layer alone; every other timed
            layer keeps the steps whose buckets overlap that window, and
            timeless layers ride along whole.
        name: Extra text folded into each cut's derived `group_id`.
        context_fn: Called once per window while the index is built; its
            result becomes that sample's `model_context`.

    Raises:
        ValueError: `root` isn't a directory, or no store under it qualifies.

    Examples:
        >>> dataset = StackDataset("data/train", layers=["image", "label"], tile_size_px=512)
        >>> dataset[0]["layers"].keys()
        dict_keys(['image', 'label'])
    """

    def __init__(
        self,
        root: str | Path,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        *,
        layers: Sequence[LayerName] | None = None,
        required_layers: Sequence[LayerName] | None = None,
        tile_size_px: int | None = None,
        stride_px: int | None = None,
        overlap: int | float | tuple[int, int] | None = None,
        mode: TilerMode = "reflect",
        vector: bool = True,
        time: TimeWindow | None = None,
        name: str | None = None,
        context_fn: ContextFn | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"StackDataset needs a directory of {STORE_SUFFIX} stores, got {self.root}")

        self.sel_bands = sel_bands
        self.dtype_override = dtype_override
        self.layers = None if layers is None else list(layers)
        self.required_layers = list(required_layers or self.layers or ())

        self.samples: list[GeoTileStack] = []
        for path in sorted(self.root.rglob(f"*{STORE_SUFFIX}")):
            if not all((path / layer).is_dir() for layer in self.required_layers):
                continue
            try:
                stack = GeoStack.open(path)
                if self.layers is not None:
                    stack = stack.select(*self.layers)
                self.samples.extend(
                    stack.tiles(
                        tile_size_px=tile_size_px,
                        stride_px=stride_px,
                        overlap=overlap,
                        mode=mode,
                        vector=vector,
                        time=time,
                        name=name,
                        context_fn=context_fn,
                    )
                )
            except (ValueError, KeyError) as error:
                warnings.warn(f"skipping {path}: {type(error).__name__}: {error}", stacklevel=2)
        if not self.samples:
            raise ValueError(
                f"no {STORE_SUFFIX} store under {self.root} carries layers {self.required_layers}"
                if self.required_layers
                else f"no {STORE_SUFFIX} store under {self.root} yielded a sample"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.root}, samples={len(self)}, layers={self.layers})"

    def __len__(self) -> int:
        """Number of samples in the index.

        Returns:
            How many samples this dataset serves.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> TensorSample:
        """Read one window into a tensor sample.

        Args:
            index: Position in the index.

        Returns:
            `{"layers": ..., "anchor": ..., "model_context": ...}`, as
            `GeoTileStack.to_tensor` builds it.

        Raises:
            KeyError: `sel_bands` or `dtype_override` names something this
                store doesn't carry.
        """
        return self.samples[index].to_sample(bands=self.sel_bands, dtype=self.dtype_override)
