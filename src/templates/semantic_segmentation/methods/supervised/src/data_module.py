from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, PreChippedGeoSampler
from torchvision import transforms

from geosave_engine.geodata.datasets import RasterImage, RasterLabel, RasterMask

Split = Literal["train", "val", "test", "predict"]
# Each split maps to one or more literal paths (no auto-appending of split name).
DataPath = dict[str, str | list[str]]


def _normalize_split_paths(data: DataPath) -> dict[str, list[Path]]:
    """Normalize each split value to a list of resolved Paths."""
    result: dict[str, list[Path]] = {}
    for split, value in data.items():
        if isinstance(value, list):
            result[split] = [Path(p) for p in value]
        else:
            result[split] = [Path(value)]
    return result


class GeosaveDataModule(LightningDataModule):
    """Semantic-segmentation datamodule over pre-chipped TorchGeo datasets.

    ``data_image``, ``data_label``, and ``data_mask`` are plain dicts mapping
    split names (``"train"``, ``"val"``, ``"test"``, ``"predict"``) to literal
    directory paths. Values may be a single path string or a list of path strings.
    Paths are used as-is — no split name is appended automatically.

    Splits absent from a dict are silently skipped (e.g. ``data_label`` with no
    ``"predict"`` key means labels are not loaded during predict).

    Batch keys: ``"image"``, ``"label"`` (if split present), ``"mask"`` (if split present).
    Normalization applied post-collation when ``mean_norm`` and ``std_norm`` are set.
    """

    def __init__(
        self,
        data_image: DataPath,
        data_label: DataPath | None = None,
        data_mask: DataPath | None = None,
        mean_norm: list[float] | None = None,
        std_norm: list[float] | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        predict_patch_size: int = 1024,
        predict_stride: int | None = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        if (mean_norm is None) != (std_norm is None):
            raise ValueError("mean_norm and std_norm must both be set or both be None")

        self.data_image = _normalize_split_paths(data_image)
        self.data_label = _normalize_split_paths(data_label) if data_label is not None else None
        self.data_mask = _normalize_split_paths(data_mask) if data_mask is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predict_patch_size = predict_patch_size
        self.predict_stride = predict_stride or predict_patch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.mean_norm = mean_norm
        self.std_norm = std_norm

    def _build_dataset(self, split: Split) -> GeoDataset:
        image_paths = self.data_image.get(split)
        if not image_paths:
            raise ValueError(f"No image paths configured for split {split!r}")

        dataset: GeoDataset = RasterImage(image_paths[0])
        for p in image_paths[1:]:
            dataset = dataset & RasterImage(p)

        if self.data_label is not None:
            label_paths = self.data_label.get(split)
            if label_paths:
                dataset = dataset & RasterLabel(label_paths[0])

        if self.data_mask is not None:
            mask_paths = self.data_mask.get(split)
            if mask_paths:
                mask_ds: GeoDataset = RasterMask(mask_paths[0])
                for p in mask_paths[1:]:
                    mask_ds = mask_ds | RasterMask(p)
                dataset = dataset & mask_ds

        return dataset

    def _collate(self, samples: list[Mapping[Any, Any]]) -> dict[Any, Any]:
        batch = stack_samples(samples)
        if self.mean_norm is not None and self.std_norm is not None:
            norm = transforms.Normalize(mean=self.mean_norm, std=self.std_norm)
            batch["image"] = norm(batch["image"])
        return batch

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")
            self.train_sampler = PreChippedGeoSampler(self.train_dataset, shuffle=True)
            self.val_sampler = PreChippedGeoSampler(self.val_dataset, shuffle=False)

        elif stage == "validate":
            self.val_dataset = self._build_dataset("val")
            self.val_sampler = PreChippedGeoSampler(self.val_dataset, shuffle=False)

        elif stage == "test":
            self.test_dataset = self._build_dataset("test")
            self.test_sampler = PreChippedGeoSampler(self.test_dataset, shuffle=False)

        elif stage == "predict":
            self.predict_dataset = self._build_dataset("predict")
            self.predict_sampler = GridGeoSampler(
                self.predict_dataset,
                size=self.predict_patch_size,
                stride=self.predict_stride,
            )

        else:
            raise ValueError(f"Invalid stage: {stage!r}")

    def _loader(self, dataset: GeoDataset, sampler) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.train_sampler)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, self.val_sampler)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, self.test_sampler)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset, self.predict_sampler)
