from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from geosave_engine.geodata.datasets import GeoDataset, GeoTileSampler, GridSampler, PreChippedSampler, stack_samples


class GeoDataModule(LightningDataModule):
    """Base datamodule for GeoDataset-backed Lightning pipelines.

    Handles dataset creation, dataloader construction, and split routing.
    Reads already-ingested data only — run ``geosave ingest`` separately to
    populate ``root`` first (see ``geosave_engine.geodata.pipeline.PipelineRunner``).

    Args:
        root: Base directory. Split subdirs read from inside.
        output_key: Layer name → batch key mapping.
            Example: ``{"sentinel_2_l1c": "image", "dynamicworld": ("label", torch.int64)}``.
        sel_bands: Per-layer band selection.
            Example: ``{"sentinel_2_l1c": ["B04", "B03", "B02"]}``.
        context_fields: GeoTile metadata fields per sample.
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
        output_key: dict[str, str | tuple[str, torch.dtype]],
        sel_bands: dict[str, list[str]] | None = None,
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
        super().__init__()
        self.root = Path(root)
        self.output_key = output_key
        self.sel_bands = sel_bands
        self.context_fields: list[str] = context_fields or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.predict_sampler_type = predict_sampler
        self.patch_size = patch_size
        self.stride = stride or patch_size

    def _make_dataset(self, root: Path, sampler: GeoTileSampler | None = None) -> GeoDataset:
        return GeoDataset(
            root,
            sampler=sampler,
            output_key=self.output_key,
            sel_bands=self.sel_bands,
            context_fields=self.context_fields,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self.train_dataset = self._make_dataset(self.root / "train")
            self.val_dataset = self._make_dataset(self.root / "val")
        elif stage == "validate":
            self.val_dataset = self._make_dataset(self.root / "val")
        elif stage == "test":
            self.test_dataset = self._make_dataset(self.root / "test")
        elif stage == "predict":
            sampler = (
                GridSampler(self.patch_size, self.stride)
                if self.predict_sampler_type == "grid"
                else PreChippedSampler()
            )
            self.predict_dataset = self._make_dataset(self.root / "predict", sampler=sampler)
        else:
            raise ValueError(f"Invalid stage: {stage!r}")

    def _loader(self, dataset: GeoDataset, *, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=stack_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset)
