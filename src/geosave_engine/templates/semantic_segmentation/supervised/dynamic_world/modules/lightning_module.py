from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

from geosave_engine.ml.callbacks.calibration import DenseCalibrationCallback
from geosave_engine.ml.core.factory import build_loss, build_optimizer, build_scheduler
from geosave_engine.ml.core.transforms import ImageAugmenter
from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.tasks import SemanticSegmentationTask
from geosave_engine.utils import colorize

# Sentinel-2 L1C band schema — matches the "sentinel_2_l1c" stac step in configs/model.yaml.
# DynamicWorld class/color schema — matches the "dynamicworld" pipeline step in configs/ingest.yaml.
# Kept as literals here (not classes) since ingestion is now config-driven, not code-driven.
_SENTINEL2_BAND_MAP = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3", "B08": "nir",
    "B09": "watervapor", "B10": "cirrus", "B11": "swir16", "B12": "swir22",
    "B8A": "nir_narrow",
}
_CLASS_MAP = {
    0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation",
    4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare",
}
_PALETTE = {
    0: "#419bdf", 1: "#397d49", 2: "#88b053", 3: "#7a87c6",
    4: "#e49635", 5: "#dfc35a", 6: "#c4281b", 7: "#a59b8f",
}


class GeosaveLightningModule(SemanticSegmentationTask):
    """DynamicWorld / Sentinel-2 segmentation task.

    Extends SemanticSegmentationTask with a default supervised training loop.
    Override training_step for custom methods (semi-supervised, UniMatchV2, etc.).

    Args:
        loss: Loss function registry key (e.g. ``"CELoss"``).
        optimizer: Optimizer registry key (e.g. ``"AdamW"``).
        scheduler: LR scheduler registry key. ``None`` disables scheduling.
        metrics: Metric names in dot notation (e.g. ``["iou.macro", "f1.macro"]``).
        augmentations: Kornia augmentation config list.
        threshold_calibration_config: kwargs forwarded to ``DenseCalibrationCallback``.
        log_image_every_n_epochs: Epoch frequency for prediction visualization logging.

    All SemanticSegmentationTask args are also accepted.
    Schema defaults (num_classes, in_channels, class_map, band_map, palette) are
    pre-filled from the schema literals above, matching configs/ingest.yaml.
    """

    def __init__(
        self,
        encoder: str | type[nn.Module] = 'dinov3',
        decoder: str | type[nn.Module] = 'dpt',
        head: str | type[nn.Module] = 'dense',
        monolith: str | type[nn.Module] | None = None,
        num_classes: int = len(_CLASS_MAP),
        in_channels: int = len(_SENTINEL2_BAND_MAP),
        input_size: int | tuple[int, int] = 224,
        ignore_index: int = 255,
        upsample_output: bool = True,
        class_map: dict | None = None,
        band_map: dict | None = None,
        palette: dict | None = None,
        mean_norm: list[float] | None = None,
        std_norm: list[float] | None = None,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
        config: dict | None = None,
        loss: str = 'CELoss',
        optimizer: str = 'AdamW',
        scheduler: str | None = None,
        metrics: list[str] | None = None,
        augmentations: list[dict] | None = None,
        threshold_calibration_config: dict | None = None,
        log_image_every_n_epochs: int = 2,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            head=head,
            monolith=monolith,
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size,
            ignore_index=ignore_index,
            upsample_output=upsample_output,
            class_map=class_map if class_map is not None else _CLASS_MAP,
            band_map=band_map if band_map is not None else _SENTINEL2_BAND_MAP,
            palette=palette if palette is not None else _PALETTE,
            mean_norm=mean_norm,
            std_norm=std_norm,
            overlap_ratio=overlap_ratio,
            pad_size=pad_size,
            config=config,
        )
        self.loss_name = loss
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.metrics_config = metrics
        self.augmentations = augmentations or []
        self.threshold_calibration_config = threshold_calibration_config or {}
        self.log_image_every_n_epochs = log_image_every_n_epochs

        self.loss_fn = build_loss(loss, {**self.config.get('loss', {}), 'ignore_index': ignore_index})

    def configure_model(self) -> None:
        super().configure_model()
        if not hasattr(self, 'augmenter'):
            self.augmenter = ImageAugmenter(augmentations=self.augmentations, size=self.input_size)

    def configure_callbacks(self):
        return [DenseCalibrationCallback(**self.threshold_calibration_config)]

    def configure_optimizers(self):
        optimizer = build_optimizer(self.optimizer_name, self.model, self.config.get('optimizer') or {})
        if self.scheduler_name is None:
            return optimizer
        scheduler = build_scheduler(self.scheduler_name, optimizer, self.config.get('scheduler') or {})
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def setup(self, stage: str | None = None) -> None:
        metrics = SemanticSegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            labels=list(self.class_map.values()) if self.class_map else None,
            metrics=self.metrics_config,
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Default supervised step. Override for semi-supervised or custom methods."""
        image, label = batch['image'], batch['label']
        image, label = self.augmenter(image, label)
        logits = self(image, **batch.get('context', {}))
        loss = self.loss_fn(logits, label)
        self.train_metrics.update(logits, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=image.shape[0])
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=image.shape[0])
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        image, label = batch['image'], batch['label']
        mask = batch.get('mask')
        logits = self(image, **batch.get('context', {}))
        loss = self.loss_fn(logits, label)
        self.val_metrics.update(logits, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if batch_idx == 0 and self.current_epoch % self.log_image_every_n_epochs == 0:
            self._log_prediction('val', logits, label, mask)

    def test_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        image, label = batch['image'], batch['label']
        mask = batch.get('mask')
        logits = self(image, **batch.get('context', {}))
        self.test_metrics.update(logits, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False)
        if batch_idx == 0 and self.current_epoch % self.log_image_every_n_epochs == 0:
            self._log_prediction('test', logits, label, mask)

    def _log_prediction(
        self,
        prefix: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if not self.palette:
            warnings.warn(
                f"{type(self).__name__}: no palette defined; skipping prediction logging.",
                UserWarning,
                stacklevel=2,
            )
            return
        preds, _ = self.postprocess(logits, mask)
        label_rgb = colorize(labels[0], self.palette)
        pred_rgb = colorize(preds[0], self.palette)
        step = self.current_epoch
        for lg in self.loggers:
            if isinstance(lg, TensorBoardLogger):
                writer = lg.experiment
                writer.add_image(f'{prefix}/label', label_rgb, step, dataformats='HWC')
                writer.add_image(f'{prefix}/prediction', pred_rgb, step, dataformats='HWC')
            elif isinstance(lg, MLFlowLogger):
                lg.experiment.log_image(lg.run_id, label_rgb, f'{prefix}_label_{step}.png')
                lg.experiment.log_image(lg.run_id, pred_rgb, f'{prefix}_prediction_{step}.png')
