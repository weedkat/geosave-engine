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
from geosave_engine.utils.geodata.manifest import (
    derive_training_meta,
    load_band_meta,
    load_class_meta,
)

# manifest.gpkg setups, changethis
CLASS_META_LAYER = "metadata_dynamicworld"
BAND_META_LAYER = "bands_sentinel_2_l1c"

# IF labels band is more than one layer, set this up, otherwise delete
DW_LABEL_BANDS: tuple[str, ...] = ("label", "prob")
RasterLabel.all_bands = DW_LABEL_BANDS


class GeosaveDataModule(LightningDataModule):
    """Semantic-segmentation datamodule over pre-chipped TorchGeo datasets.

    Layout driven by ``data_path``:
      - fit/validate/test: ``<data_path>/{sentinel_2_l1c,dynamicworld,cloud_mask}/{train,val,test}``
      - predict:           ``<data_path>/sentinel_2_l1c`` (COGs)

    Training samples intersect ``sentinel_2_l1c & dynamicworld & cloud_mask``;
    the dynamicworld mask is exposed as ``sample["label"]`` and the cloud mask as
    ``sample["cloud_mask"]``. ``num_classes``, ``ignore_index``, ``class_id_map``,
    and ``in_channels`` are derived from the manifest's metadata layers.
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

        # Manifest metadata is read up-front so the LightningModule can read
        # `in_channels` / `training_meta` from `configure_model()` without
        # having to wait for `setup()` to run on every stage.
        class_meta = load_class_meta(self.manifest_path, layer=CLASS_META_LAYER)
        band_meta = load_band_meta(self.manifest_path, layer=BAND_META_LAYER)
        self.training_meta = derive_training_meta(class_meta)
        self.bands: list[str] = band_meta["band_name"].astype(str).tolist()
        self.in_channels: int = len(self.bands)

    def setup(self, stage: str | None = None) -> None:
        # The dw label is the primary "mask" through augmentation, then renamed
        # to "label" so the LightningModule reads `batch["label"]`.
        rename_label = rename_key("mask", "label")

        train_pipeline = TransformPipeline(
            remap(self.training_meta.class_id_map),
            TransformsCompose(self.train_transform),
            rename_label,
        )
        infer_pipeline = TransformPipeline(
            remap(self.training_meta.class_id_map),
            TransformsCompose(self.infer_transform),
            rename_label,
        )
        # Predict-time samples have no label, so skip the mask remap/rename.
        predict_pipeline = TransformPipeline(
            TransformsCompose(self.infer_transform),
        )

        # Override the Sentinel2L1C class-level `all_bands` so band indexing
        # matches the manifest-declared layout (the ingestion writes only the
        # subset of bands listed in the manifest, in that order).
        sentinel_cls = type(
            "Sentinel2L1CSubset",
            (Sentinel2L1C,),
            {"all_bands": tuple(self.bands)},
        )

        # NOTE: cloud_mask is intentionally not intersected here. The default
        # `TransformsCompose` only augments `image` + `mask`, so any extra
        # raster (cloud_mask) would skip augmentation and break batch stacking.
        # Wire it back in once the augmentation pipeline supports additional
        # mask targets.

        def _dw(split: str) -> RasterLabel:
            return RasterLabel(
                paths=self.data_path / "dynamicworld" / split,
                bands=("label",),
            )
        
        def _s2(split: str) -> Sentinel2L1C:
            return sentinel_cls(
                paths=self.data_path / "sentinel_2_l1c" / split,
            )

        if stage == "fit":
            self.train_dataset = _s2("train") & _dw("train")
            self.train_dataset.transforms = train_pipeline
            self.train_sampler = PreChippedGeoSampler(self.train_dataset, shuffle=True)

            self.val_dataset = _s2("val") & _dw("val")
            self.val_dataset.transforms = infer_pipeline
            self.val_sampler = PreChippedGeoSampler(self.val_dataset, shuffle=False)

        elif stage == "validate":
            self.val_dataset = _s2("val") & _dw("val")
            self.val_dataset.transforms = infer_pipeline
            self.val_sampler = PreChippedGeoSampler(self.val_dataset, shuffle=False)

        elif stage == "test":
            self.test_dataset = _s2("test") & _dw("test")
            self.test_dataset.transforms = infer_pipeline
            self.test_sampler = PreChippedGeoSampler(self.test_dataset, shuffle=False)

        elif stage == "predict":
            self.predict_dataset = sentinel_cls(paths=self.data_path / "sentinel_2_l1c")
            self.predict_dataset.transforms = predict_pipeline
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
