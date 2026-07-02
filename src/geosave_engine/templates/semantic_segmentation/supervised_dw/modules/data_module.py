from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from geosave_engine.geodata.core import GeoTile
from geosave_engine.geodata.datasets import GeoDataset, GridSampler, PreChippedSampler, stack_samples

from .data_pipeline import (
    predict_pipeline,
    training_pipeline,
)

class DynamicWorldDataset(GeoDataset):
    output_key = {
        "sentinel_2_l1c": "image",
        "cloud_mask":      ("mask",  torch.bool),
        "ndvi":            ("ndvi",  torch.float32),
        "dynamicworld":    ("label", torch.int64),
    }

    def context(self, tiles: dict[str, GeoTile]) -> dict:
        """Add spatial and temporal metadata to each batch sample.

        Returns:
            {
                "crs": str,
                "transform": Affine,
                "coordinate": tuple[float, float],
                "time": int,  # day of year (1–365)
            }.
        """
        ref_tile = next(iter(tiles.values()))
        return {
            "crs": ref_tile.crs,
            "transform": ref_tile.affine,
            "coordinate": ref_tile.centroid,
            "time": ref_tile.datetime.timetuple().tm_yday,
        }


class DynamicWorldDatasetRGB(DynamicWorldDataset):
    """RGB-only variant — loads B04/B03/B02 instead of all S2 bands."""

    sel_bands = {
        "sentinel_2_l1c": ["B04", "B03", "B02"],
    }


class GeosaveDataModule(LightningDataModule):
    """Semantic-segmentation datamodule for Sentinel-2 / DynamicWorld.

    Creates one pipeline set per split (train/val/test/predict) rooted at
    ``<root>/<split>/``. Ingestion runs in ``prepare_data`` only when
    ``ingest=True``.

    Args:
        root: Base directory. Split subdirs are created automatically.
        split_dirs: Map of split → GeoTIFF source directory for training ingestion.
            Example: ``{"train": "data/raw/train/", "val": "data/raw/val/"}``.
        predict_sources: List of ingest source dicts for the predict stage.
            Each entry must include a ``"type"`` discriminator key.
            Example: ``[{"type": "geojson", "src": "aoi.geojson", "datetime": "2024-06-01"}]``.
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
        split_dirs: dict[str, str] | None = None,
        predict_sources: list[dict] | None = None,
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
        super().__init__()
        self.root = Path(root)
        self.split_dirs: dict[str, str] = split_dirs or {}
        self.predict_sources: list[dict] = predict_sources or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.predict_sampler_type = predict_sampler
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.ingest = ingest
        self.max_tiles = max_tiles

    def _dataset_cls(self) -> type[GeoDataset]:
        return DynamicWorldDatasetRGB

    def prepare_data(self) -> None:
        if not self.ingest:
            return

        if self.predict_sources:
            predict_pipeline(self.root / "predict", self.predict_sources, max_item=self.max_tiles)
            return

        for split, dir_ in self.split_dirs.items():
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split in split_dirs: {split!r}")
            training_pipeline(self.root / split, dir_, max_item=self.max_tiles)

    def setup(self, stage: str | None = None) -> None:
        cls = self._dataset_cls()
        if stage == "fit":
            self.train_dataset = cls(self.root / "train")
            self.val_dataset = cls(self.root / "val")
        elif stage == "validate":
            self.val_dataset = cls(self.root / "val")
        elif stage == "test":
            self.test_dataset = cls(self.root / "test")
        elif stage == "predict":
            if self.predict_sampler_type == "grid":
                sampler = GridSampler(self.patch_size, self.stride)
            else:
                sampler = PreChippedSampler()
            self.predict_dataset = cls(self.root / "predict", sampler=sampler)
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
