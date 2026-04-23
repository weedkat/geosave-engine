from __future__ import annotations

from os import PathLike
from pathlib import Path

import geopandas as gpd
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples

from src.dataset_train import GeosaveDataset
from geosave_engine.utils.geodata.manifest import load_class_meta
from geosave_engine.ml.inference import GeoPredictRasterDataset, build_grid_sampler
from geosave_engine.ml.core.transform import TransformsCompose


class GeosaveDataModule(LightningDataModule):
    """Data module driven by a GeoPackage manifest.

    The manifest must have columns: ``split`` ("train"/"val"/"test"),
    ``input_path``, and ``label_path``.
    """

    def __init__(
        self,
        manifest: str,
        predict_paths: str | PathLike[str] | list[str | PathLike[str]] | None = None,
        predict_patch_size: int | tuple[int, int] = 256,
        predict_stride: int | tuple[int, int] | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        train_transform: list | None = None,
        infer_transform: list | None = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.manifest = manifest
        self.predict_paths = predict_paths
        self.predict_patch_size = predict_patch_size
        self.predict_stride = predict_stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = TransformsCompose(train_transform) if train_transform else None
        self.infer_transform = TransformsCompose(infer_transform) if infer_transform else None
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.predict_sampler = None

    def _load_split(self, split: str) -> list[tuple[str, str, str]]:
        gdf = gpd.read_file(self.manifest, layer="manifest", engine="pyogrio")
        rows = gdf[gdf["split"] == split]
        return list(zip(rows["input_path"], rows["label_path"], rows["mask_path"]))

    def setup(self, stage: str | None = None) -> None:
        self.class_meta = load_class_meta(Path(self.manifest))

        if stage in ("fit", "validate", "test"):
            self.train_dataset = GeosaveDataset(self._load_split("train"), class_meta=self.class_meta, transform=self.train_transform)
            self.val_dataset   = GeosaveDataset(self._load_split("val"),   class_meta=self.class_meta, transform=self.infer_transform)
            self.test_dataset  = GeosaveDataset(self._load_split("test"),  class_meta=self.class_meta, transform=self.infer_transform)

        if stage == "predict":
            if self.predict_paths is None:
                raise ValueError("predict_paths is required for predict stage")

            self.predict_dataset = GeoPredictRasterDataset(
                paths=self.predict_paths,
                bands=self.predict_bands,
            )
            self.predict_sampler = build_grid_sampler(
                self.predict_dataset,
                patch_size=self.predict_patch_size,
                stride=self.predict_stride,
            )

    def _make_dataloader(self, dataset: GeosaveDataset, *, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        if self.predict_sampler is None:
            raise RuntimeError("predict sampler is not initialized; call setup('predict') first")

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.predict_sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=stack_samples,
        )
