from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from geosave_engine.geodata.datasets import GeoDataset, GridSampler, PreChippedSampler, stack_samples

from .data_pipeline import DataPipeline, DynamicWorldDatasetRGB, LabelPipeline


class GeosaveDataModule(LightningDataModule):
    """Semantic-segmentation datamodule for Sentinel-2 imagery.

    Creates one ``DataPipeline`` and one ``LabelPipeline`` per split (train/val/test)
    rooted at ``<root>/<split>/``. Ingestion is optional — omit ``ingest_dirs``
    if data is already on disk.

    Args:
        root: Base directory. Split subdirs are derived automatically:
            ``root/train/``, ``root/val/``, ``root/test/``.
        ingest_dirs: Optional map of split → anchor tiff directory.
            Triggers ingestion in ``prepare_data`` for each key.
            Example: ``{"train": "data/raw/train/", "val": "data/raw/val/"}``.
        ingest_geojsons: Optional list of GeoJSON ingest configs for predict.
            Each entry: ``{"src": "path/to/file.geojson", "datetime": "..."}``.
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
        ingest_dirs: dict[str, str] | None = None,
        ingest_geojsons: list[dict] | None = None,
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
        self.ingest_dirs: dict[str, str] = ingest_dirs or {}
        self.ingest_geojsons = ingest_geojsons or []
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

        self.predict_pipeline = DataPipeline(self.root / "predict")

    def _dataset_cls(self):
        return DynamicWorldDatasetRGB

    def prepare_data(self) -> None:
        if self.ingest_geojsons:
            for entry in self.ingest_geojsons:
                self.predict_pipeline.ingest_from_geojson(**entry)
            return

        if self.ingest:
            for split, dir_ in self.ingest_dirs.items():
                if split not in {"train", "val", "test"}:
                    raise ValueError(f"Invalid split in ingest_dirs: {split!r}")
                root = self.root / split
                DataPipeline(root).ingest_from_geotiff(dir_, max_item=self.max_tiles)
                LabelPipeline(root).ingest_from_geotiff(dir_, max_item=self.max_tiles)

    def setup(self, stage: str | None = None) -> None:
        cls = self._dataset_cls()
        if stage == "fit":
            self.train_dataset = cls.from_dir(self.root / "train")
            self.val_dataset = cls.from_dir(self.root / "val")
        elif stage == "validate":
            self.val_dataset = cls.from_dir(self.root / "val")
        elif stage == "test":
            self.test_dataset = cls.from_dir(self.root / "test")
        elif stage == "predict":
            if self.predict_sampler_type == "grid":
                sampler = GridSampler(self.patch_size, self.stride)
            else:
                sampler = PreChippedSampler()
            self.predict_dataset = cls.from_dir(self.root, sampler=sampler)
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
