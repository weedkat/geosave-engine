from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from geosave_engine.geodata.core import source_from_dict
from geosave_engine.ml.data import GeoDataModule

from modules.pipeline import ImagePipeline, LabelPipeline

_DEFAULT_OUTPUT_KEY: dict[str, str | tuple[str, torch.dtype]] = {
    "image": "image",
    "label": ("label", torch.int64),
}


class GeosaveDataModule(GeoDataModule):
    """Segmentation datamodule — customize for your catalog.

    Add your pipeline classes to ``prepare_data`` and update
    ``class_map``, ``band_map``, ``palette`` properties to match.

    Args:
        root: Base directory. Split subdirs created inside.
        sources: Map of split name → source config dict.
        context_fields: GeoTile metadata fields per sample. Defaults to none.
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
        ingest: Run ingestion in ``prepare_data`` when ``True``.
        max_tiles: Stop ingestion after this many tiles. ``None`` processes all.
    """

    def __init__(
        self,
        root: str | Path,
        sources: dict[str, dict] | None = None,
        context_fields: list[str] | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        predict_sampler: Literal["prechipped", "grid"] = "prechipped",
        patch_size: int = 1024,
        stride: int | None = None,
        ingest: bool = False,
        max_tiles: int | None = None,
    ) -> None:
        super().__init__(
            root=root,
            output_key=_DEFAULT_OUTPUT_KEY,
            sources=sources,
            context_fields=context_fields,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            predict_sampler=predict_sampler,
            patch_size=patch_size,
            stride=stride,
            ingest=ingest,
            max_tiles=max_tiles,
        )

    @property
    def class_map(self) -> dict[int, str]:
        return LabelPipeline.class_map()

    @property
    def band_map(self) -> dict[str, int]:
        return ImagePipeline.band_map()

    @property
    def palette(self) -> dict[int, str]:
        return LabelPipeline.color_map()

    def prepare_data(self) -> None:
        if not self.ingest:
            return
        for split, src_dict in self.sources.items():
            source = source_from_dict(src_dict)
            split_root = self.root / split
            ImagePipeline(split_root).ingest_from(source, max_item=self.max_tiles)
            if split != "predict":
                LabelPipeline(split_root).ingest_from(source, max_item=self.max_tiles)
