from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from geosave_engine.ml.data import GeoDataModule

_OUTPUT_KEY: dict[str, str | tuple[str, torch.dtype]] = {
    "image": "image",
    "label": ("label", torch.int64),
}
_ALL_CONTEXT_FIELDS: list[str] = [
    "crs", "transform", "coordinate", "time", "datetime", "bbox_wgs84", "stac_item_ids",
]


class GeosaveDataModule(GeoDataModule):
    """Segmentation datamodule — customize ``_OUTPUT_KEY`` for your layer names.

    Reads already-ingested data. Run ``geosave ingest -c configs/ingest.yaml``
    separately to populate ``root`` first.

    Args:
        root: Base directory. Split subdirs read from inside.
        context_fields: GeoTile metadata fields per sample. Defaults to all.
            Valid: ``crs``, ``transform``, ``coordinate``, ``time``,
            ``datetime``, ``bbox_wgs84``, ``stac_item_ids``.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.
        predict_sampler: Sampler strategy for predict stage.
        patch_size: Spatial patch size in pixels (grid sampler only).
        stride: Stride between patches. Defaults to ``patch_size``.
    """

    def __init__(
        self,
        root: str | Path,
        context_fields: list[str] | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        predict_sampler: Literal["prechipped", "grid"] = "prechipped",
        patch_size: int = 1024,
        stride: int | None = None,
    ) -> None:
        super().__init__(
            root=root,
            output_key=_OUTPUT_KEY,
            context_fields=context_fields if context_fields is not None else _ALL_CONTEXT_FIELDS,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            predict_sampler=predict_sampler,
            patch_size=patch_size,
            stride=stride,
        )
