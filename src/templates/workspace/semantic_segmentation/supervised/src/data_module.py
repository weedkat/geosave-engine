from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler, PreChippedGeoSampler

from geosave_engine.geodata.datasets import RasterLabel, Sentinel2L1C
from geosave_engine.ml.core.transform import (
    TransformPipeline,
    TransformsCompose,
    remap,
    rename_key,
)
from geosave_engine.utils.geodata.manifest import load_class_meta

# Replace with the source-class-id -> training-class-id mapping for your label
# product. Keys must cover every value that appears in the raster labels.
CLASS_ID_MAP: dict[int, int] = {
    # 0: 255,  # e.g. no-data -> ignore
    # 1: 0,
    # 2: 1,
}

class GeosaveDataModule(LightningDataModule):
    """Semantic-segmentation datamodule over pre-chipped TorchGeo datasets.

    Layout driven by ``data_path``:
      - fit/validate/test: ``<data_path>/{sentinel_2_l1c,dynamicworld,cloud_mask}/{train,val,test}``
      - predict:           ``<data_path>/sentinel_2_l1c`` (COGs)

    Training samples intersect ``sentinel_2_l1c & dynamicworld & cloud_mask``;
    the cloud mask is available at ``sample["cloud_mask"]``.
    """

    def __init__(
        self,
        data_path: str,
        manifest_path: str,
        train_transform: list,
        infer_transform: list,
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
        self.data_path = Path(data_path)
        self.manifest_path = Path(manifest_path)
        self.train_transform = train_transform
        self.infer_transform = infer_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predict_patch_size = predict_patch_size
        self.predict_stride = predict_stride if predict_stride is not None else predict_patch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def setup(self, stage: str | None = None) -> None:
        self.class_meta = load_class_meta(self.manifest_path)

        train_pipeline = TransformPipeline(
            remap(CLASS_ID_MAP),
            TransformsCompose(self.train_transform),
        )
        infer_pipeline = TransformPipeline(
            remap(CLASS_ID_MAP),
            TransformsCompose(self.infer_transform),
        )
        # so we can use batch["image"], batch["mask"], batch["cloud_mask"] in the model
        rename_cloud = rename_key("mask", "cloud_mask") 

        if stage == "fit":
            s2_train = Sentinel2L1C(paths=self.data_path / "sentinel_2_l1c" / "train")
            dw_train = RasterLabel(paths=self.data_path / "dynamicworld" / "train")
            cm_train = RasterLabel(paths=self.data_path / "cloud_mask" / "train", transforms=rename_cloud)
            self.train_dataset = s2_train & dw_train & cm_train
            self.train_dataset.transforms = train_pipeline
            self.train_sampler = PreChippedGeoSampler(self.train_dataset, shuffle=True)
        
        elif stage == "validate":
            s2_val = Sentinel2L1C(paths=self.data_path / "sentinel_2_l1c" / "val")
            dw_val = RasterLabel(paths=self.data_path / "dynamicworld" / "val")
            cm_val = RasterLabel(paths=self.data_path / "cloud_mask" / "val", transforms=rename_cloud)
            self.val_dataset = s2_val & dw_val & cm_val
            self.val_dataset.transforms = infer_pipeline
            self.val_sampler = PreChippedGeoSampler(self.val_dataset, shuffle=False)

        elif stage == "test":
            s2_test = Sentinel2L1C(paths=self.data_path / "sentinel_2_l1c" / "test")
            dw_test = RasterLabel(paths=self.data_path / "dynamicworld" / "test")
            cm_test = RasterLabel(paths=self.data_path / "cloud_mask" / "test", transforms=rename_cloud)
            self.test_dataset = s2_test & dw_test & cm_test
            self.test_dataset.transforms = infer_pipeline
            self.test_sampler = PreChippedGeoSampler(self.test_dataset, shuffle=False)

        elif stage == "predict":
            self.predict_dataset = Sentinel2L1C(paths=self.data_path / "sentinel_2_l1c")
            self.predict_dataset.transforms = infer_pipeline
            self.predict_sampler = GridGeoSampler(
                self.predict_dataset,
                size=self.predict_patch_size,
                stride=self.predict_stride,
            )
        
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _loader(self, dataset, sampler) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=stack_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.train_sampler)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, self.val_sampler)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, self.test_sampler)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset, self.predict_sampler)
