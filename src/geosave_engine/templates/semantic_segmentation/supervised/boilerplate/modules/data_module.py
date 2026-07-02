from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from geosave_engine.geodata.core import source_from_dict
from geosave_engine.geodata.datasets import GeoDataset, GridSampler, PreChippedSampler, stack_samples

from modules.pipeline import ImagePipeline, LabelPipeline
from modules.dataset import WorkspaceDataset


class GeosaveDataModule(LightningDataModule):
    """Segmentation datamodule — customize for your catalog.

    Add your pipeline classes to ``prepare_data`` and update
    ``class_map``, ``band_map``, ``palette`` properties to match.

    Args:
        root: Base directory. Split subdirs created inside.
        sources: Map of split name → source config dict.
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
        self.sources: dict[str, dict] = sources or {}
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

    # ------------------------------------------------------------------
    # Schema metadata — consumed by GeosaveLightningModule.setup()
    # ------------------------------------------------------------------

    @property
    def class_map(self) -> dict[int, str]:
        return LabelPipeline.class_map()

    @property
    def band_map(self) -> dict[str, int]:
        return ImagePipeline.band_map()

    @property
    def palette(self) -> dict[int, str]:
        return LabelPipeline.color_map()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        if not self.ingest:
            return
        for split, src_dict in self.sources.items():
            source = source_from_dict(src_dict)
            split_root = self.root / split
            ImagePipeline(split_root).ingest_from(source, max_item=self.max_tiles)
            if split != "predict":
                LabelPipeline(split_root).ingest_from(source, max_item=self.max_tiles)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self.train_dataset = WorkspaceDataset(self.root / "train")
            self.val_dataset = WorkspaceDataset(self.root / "val")
        elif stage == "validate":
            self.val_dataset = WorkspaceDataset(self.root / "val")
        elif stage == "test":
            self.test_dataset = WorkspaceDataset(self.root / "test")
        elif stage == "predict":
            sampler = (
                GridSampler(self.patch_size, self.stride)
                if self.predict_sampler_type == "grid"
                else PreChippedSampler()
            )
            self.predict_dataset = WorkspaceDataset(self.root / "predict", sampler=sampler)
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
