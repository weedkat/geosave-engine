from __future__ import annotations

import math

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import random_grid_cell_assignment, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

from dataset import GeosaveDataset

# https://torchgeo.readthedocs.io/en/v0.5.0/_modules/torchgeo/datasets/splits.html
# https://torchgeo.readthedocs.io/en/latest/api/samplers.html


class GeosaveDataModule(LightningDataModule):
    """Base class for GeoSave data modules."""

    def __init__(
        self,
        input_data_dir: str,
        label_data_dir: str | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        patch_size: int | tuple[int, int] = 256,
        stride: int | tuple[int, int] | None = 128,
        split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
        grid_size: int = 6,
        train_length: int | None = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ):
        if not math.isclose(sum(split_fractions), 1.0):
            raise ValueError("split_fractions must sum to 1.0")

        super().__init__()
        self.input_data_dir = input_data_dir
        self.label_data_dir = label_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride
        self.split_fractions = split_fractions
        self.grid_size = grid_size
        self.train_length = train_length
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        """Download or index data before distributed setup begins."""
        return None

    def _build_image_dataset(self) -> GeosaveDataset:
        return GeosaveDataset(data_dir=self.input_data_dir)

    def _build_label_dataset(self) -> GeosaveDataset:
        if self.label_data_dir is None:
            raise ValueError("label_data_dir is required for fit, validate, and test stages")

        return GeosaveDataset(data_dir=self.label_data_dir)

    def _split_train_val_test(self, dataset):
        return random_grid_cell_assignment(
            dataset,
            self.split_fractions,
            grid_size=self.grid_size,
            generator=torch.Generator().manual_seed(0),
        )

    def _setup_training_datasets(self):
        image_dataset = self._build_image_dataset()
        label_dataset = self._build_label_dataset()
        dataset = image_dataset & label_dataset
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_train_val_test(dataset)

    def _setup_predict_dataset(self):
        self.predict_dataset = self._build_image_dataset()

    def setup(self, stage: str | None = None):
        """Create stage-specific datasets and patch samplers."""
        if stage is None:
            return None

        if stage not in {"fit", "validate", "test", "predict"}:
            raise ValueError(
                f"Unknown stage: {stage}. Expected one of 'fit', 'validate', 'test', or 'predict'."
            )

        if stage in {"fit", "validate", "test"}:
            self._setup_training_datasets()

        if stage == "predict":
            self._setup_predict_dataset()

        return None

    def _make_dataloader(self, dataset, sampler):
        """Create a DataLoader with shared configuration."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=stack_samples,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def train_dataloader(self):
        sampler = RandomGeoSampler(
            self.train_dataset,
            size=self.patch_size,
            length=self.train_length,
        )
        return self._make_dataloader(self.train_dataset, sampler)

    def val_dataloader(self):
        sampler = GridGeoSampler(
            self.val_dataset,
            size=self.patch_size,
            stride=self.stride,
        )
        return self._make_dataloader(self.val_dataset, sampler)

    def test_dataloader(self):
        sampler = GridGeoSampler(
            self.test_dataset,
            size=self.patch_size,
            stride=self.stride,
        )
        return self._make_dataloader(self.test_dataset, sampler)

    def predict_dataloader(self):
        sampler = GridGeoSampler(
            self.predict_dataset,
            size=self.patch_size,
            stride=self.stride,
        )
        return self._make_dataloader(self.predict_dataset, sampler)
