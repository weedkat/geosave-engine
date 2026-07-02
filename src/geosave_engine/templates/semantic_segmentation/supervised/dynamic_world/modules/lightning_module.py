from __future__ import annotations
import warnings

from typing import cast

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

from geosave_engine.utils import colorize
from geosave_engine.ml.callbacks import DenseCalibrationCallback
from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.core.transforms import ImageAugmenter, ImageProcessor
from geosave_engine.ml.core.factory import build_loss, build_optimizer, build_scheduler
from geosave_engine.ml.models.task.segmentation import SemanticSegmentation

from modules.pipeline import LabelPipeline, Sentinel2Pipeline


class GeosaveLightningModule(LightningModule):
    """Semantic-segmentation Lightning module for DynamicWorld / Sentinel-2.

    ``num_classes`` and ``in_channels`` default to pipeline schema counts.
    ``class_map``, ``band_map``, ``palette`` are derived from pipeline classes.

    Panel rendering and live progress display are owned by
    :class:`geosave_engine.ml.callbacks.LiveTrainingMonitor`. The module should
    log any metrics it wants displayed via standard Lightning ``self.log`` keys.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        in_channels: int | None = None,
        encoder: str = 'dinov3',
        decoder: str = 'dpt',
        head: str = 'dense',
        optimizer: str = 'AdamW',
        scheduler: str | None = None,
        loss: str = 'CELoss',
        config: dict | None = None,
        input_size: tuple[int, int] | int | None = 224,
        ignore_index: int = 255,
        metrics: list[str] | None = None,
        mean_norm: list[float] | None = None,
        std_norm: list[float] | None = None,
        augmentations: list[dict] | None = None,
        threshold_calibration_config: dict | None = None,
        log_image_every_n_epochs: int = 2,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
    ):
        if num_classes is None:
            num_classes = len(LabelPipeline.schema)
        if in_channels is None:
            in_channels = len(Sentinel2Pipeline.bands)

        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.config = config or {}
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.ignore_index = ignore_index
        self.class_map = LabelPipeline.class_map()
        self.band_map = Sentinel2Pipeline.band_map()
        self.palette = LabelPipeline.color_map()
        self.metrics = metrics or []
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.augmentations = augmentations
        self.threshold_calibration_config = threshold_calibration_config
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.overlap_ratio = overlap_ratio
        self.pad_size = pad_size

        self.loss_fn = build_loss(loss, {**self.config.get('loss', {}), "ignore_index": self.ignore_index})

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------
    def configure_model(self) -> None:
        if hasattr(self, "model"):
            return  # idempotent — Lightning may call this more than once

        self.model = SemanticSegmentation(
            encoder=self.encoder,
            decoder=self.decoder,
            head=self.head,
            encoder_config=self.config.get('encoder'),
            decoder_config=self.config.get('decoder'),
            head_config=self.config.get('head'),
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            input_size=self.input_size,
        )
        self.augmenter = ImageAugmenter(augmentations=self.augmentations or [], size=self.input_size)
        self.preprocessor = ImageProcessor(
            model=self.model.encoder,
            mean_norm=self.mean_norm,
            std_norm=self.std_norm,
        )

        self.register_buffer(
            "class_thresholds",
            torch.full((self.num_classes,), 0.5),
        )

    def configure_callbacks(self):
        config = self.threshold_calibration_config or {}
        return [DenseCalibrationCallback(**config)]

    def configure_optimizers(self):
        optimizer_name: str = self.optimizer
        scheduler_name: str | None = self.scheduler
        optimizer_config: dict = self.config.get('optimizer') or {}
        scheduler_config: dict = self.config.get('scheduler') or {}

        optimizer = build_optimizer(optimizer_name, self.model, optimizer_config)
        if scheduler_name is None:
            return optimizer
        scheduler = build_scheduler(scheduler_name, optimizer, scheduler_config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage: str | None = None) -> None:
        metrics = SemanticSegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            labels=list(self.class_map.values()) if self.class_map else None,
            metrics=self.metrics,
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self,
        image: torch.Tensor,
        crs: list[str] | None = None,
        coordinate: list[tuple[float, float]] | None = None,
        transform: list | None = None,
        time: list | None = None,
    ) -> torch.Tensor:
        """Preprocess image then run model.

        Training: direct model call (patches already at ``input_size``).
        Eval/predict: sliding-window inference with Hann blending.

        Augmentation is NOT done here — call ``self.augmenter`` in ``training_step``
        before invoking ``forward`` so label and mask stay spatially aligned.

        Args:
            image: Float image tensor ``[B, C, H, W]``.
            crs: Per-sample CRS strings. Length = B.
            coordinate: Per-sample centroid coordinates in (lon, lat) order. Length = B.
            transform: Per-sample affine transforms. Length = B.
            time: Per-sample timestamps or labels. Length = B.
        """
        image = self.preprocessor(image)
        sample_meta = dict(crs=crs, coordinate=coordinate, transform=transform, time=time)
        if self.training:
            return self.model(image, **sample_meta)
        return self.model.forward_sliding(
            image,
            overlap_ratio=self.overlap_ratio,
            pad_size=self.pad_size,
            **sample_meta,
        )

    # ---------------------------------------------------------------------
    # Training / validation / test / predict
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        context = batch.get("context") or {}
        batch_size = image.shape[0]

        image, label = self.augmenter(image, label)
        logits = self(image, **context)
        loss = self.loss_fn(logits, label)

        self.train_metrics.update(logits, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        mask = batch.get("mask")
        context = batch.get("context") or {}

        logits = self(image, **context)
        loss = self.loss_fn(logits, label)

        self.val_metrics.update(logits, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        if batch_idx == 0 and (self.current_epoch % self.log_image_every_n_epochs == 0):
            self._log_prediction("val", logits, label, mask)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        mask = batch.get("mask")
        context = batch.get("context") or {}

        logits = self(image, **context)

        if batch_idx == 0 and (self.current_epoch % self.log_image_every_n_epochs == 0):
            self._log_prediction("test", logits, label, mask)

        self.test_metrics.update(logits, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch["image"]
        mask = batch.get("mask")
        context = batch.get("context") or {}

        logits = self(image, **context)
        preds, max_probs = self.postprocess(logits, mask)
        return preds, max_probs

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _log_prediction(
        self,
        prefix: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if self.palette is None:
            warnings.warn("No palette defined in LabelPipeline.schema; skipping prediction logging.")
            return

        preds, _ = self.postprocess(logits, mask)

        label_rgb = colorize(labels[0], self.palette)
        pred_rgb = colorize(preds[0], self.palette)

        step = self.current_epoch
        for lg in self.loggers:
            if isinstance(lg, TensorBoardLogger):
                writer = lg.experiment
                writer.add_image(f"{prefix}/label", label_rgb, step, dataformats="HWC")
                writer.add_image(f"{prefix}/prediction", pred_rgb, step, dataformats="HWC")
            elif isinstance(lg, MLFlowLogger):
                lg.experiment.log_image(lg.run_id, label_rgb, f"{prefix}_label_{step}.png")
                lg.experiment.log_image(lg.run_id, pred_rgb, f"{prefix}_prediction_{step}.png")

    def postprocess(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        probs = logits.softmax(dim=1)
        max_probs, preds = probs.max(dim=1)

        class_thresholds = cast(torch.Tensor, self.class_thresholds)
        pixel_thresholds = torch.index_select(class_thresholds, 0, preds.reshape(-1)).view_as(preds)
        preds = torch.where(
            max_probs >= pixel_thresholds,
            preds,
            preds.new_full((), self.ignore_index),
        )

        if mask is not None:
            preds = torch.where(mask.bool(), preds.new_full((), self.ignore_index), preds)

        return preds, max_probs
