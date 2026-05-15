from __future__ import annotations
import warnings

from typing import cast

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

from geosave_engine.utils import colorize
from geosave_engine.ml.core.transforms import SemanticSegmentationWrapper
from geosave_engine.ml.callbacks import CalibrationCallback
from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.inference.sliding_window import infer_sliding_window

from .factory import build_loss, build_model, build_optimizer, build_scheduler 

class GeosaveLightningModule(LightningModule):
    """Semantic-segmentation Lightning module driven by manifest metadata.

    ``in_channels``, ``num_classes``, ``ignore_index`` are pulled from
    ``self.trainer.datamodule`` (via ``training_meta`` + ``in_channels``) at
    ``configure_model`` time, so the YAML config does not need to repeat them.

    Panel rendering and live progress display are owned by
    :class:`geosave_engine.ml.callbacks.LiveTrainingMonitor`. The module should
    log any metrics it wants displayed via standard Lightning ``self.log`` keys.
    """

    def __init__(
        self,
        model_config: dict,
        optim_config: dict,
        loss_config: dict,
        input_size: tuple[int, int] | int,
        num_classes: int,
        in_channels: int,
        ignore_index: int = 255,
        threshold_calibration_config: dict | None = None,
        augmentations: list[dict] | None = None,
        metrics: list[str] | None = None,
        class_map: dict[int, dict] | None = None,
        band_map: dict[int, dict] | None = None,
        log_image_every_n_epochs: int = 2,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optim_config = optim_config
        self.loss_config = loss_config
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.ignore_index = ignore_index
        self.augmentations = augmentations
        self.metrics = metrics
        self.threshold_calibration_config = threshold_calibration_config
        self.class_map = class_map
        self.band_map = band_map
        self.overlap_ratio = overlap_ratio
        self.pad_size = pad_size

        self.log_image_every_n_epochs = log_image_every_n_epochs

        self.palette = {idx: entry["color"] for idx, entry in self.class_map.items()} if self.class_map else None

        self.loss = build_loss(config={**self.loss_config, "ignore_index": self.ignore_index})
        
    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------
    def configure_model(self) -> None:
        if hasattr(self, "model"):
            return  # idempotent — Lightning may call this more than once

        self.model_config['init_args'].update(
            num_classes=self.num_classes,
            in_channels=self.in_channels,
        )
        model = build_model(self.model_config)
        self.model = SemanticSegmentationWrapper(
            model=model,
            augmentations=self.augmentations,
            size=self.input_size,
        )
        
        self.register_buffer(
            "class_thresholds",
            torch.full((self.num_classes,), 0.5),
        )

    def configure_callbacks(self):
        config = self.threshold_calibration_config or {}
        return [CalibrationCallback(**config)]

    def configure_optimizers(self):
        optimizer = build_optimizer(self.optim_config, self.model)
        scheduler = build_scheduler(self.optim_config.get("scheduler"), optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler} if scheduler else optimizer

    def setup(self, stage: str | None = None) -> None:
        metrics = SemanticSegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            labels=[c["name"] for c in self.class_map.values()] if self.class_map else None,
            metrics=self.metrics,
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    # ---------------------------------------------------------------------
    # Training / validation / test / predict
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        image, label= batch["image"], batch["label"]
        mask = batch.get("mask")
        if mask is not None:
            label = label & mask
        
        logits, label = self.model(image, label)
        loss = self.loss(logits, label)

        self.train_metrics.update(logits, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        mask = batch.get("mask")
        if mask is not None:
            label = label & mask

        logits = self._infer_sliding_window(image)
        loss = self.loss(logits, label)

        self.val_metrics.update(logits, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        if batch_idx == 0 and (self.current_epoch % self.log_image_every_n_epochs == 0):
            self._log_prediction("val", logits, label, mask)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        mask = batch.get("mask")
        if mask is not None:
            label = label & mask

        logits = self._infer_sliding_window(image)

        if batch_idx == 0 and (self.current_epoch % self.log_image_every_n_epochs == 0):
            self._log_prediction("test", logits, label, mask)
                
        self.test_metrics.update(logits, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch["image"]
        mask = batch.get("mask")
        
        logits = self._infer_sliding_window(image)
        preds, max_probs = self.postprocess(logits, mask)
        return preds, max_probs

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _infer_sliding_window(self, image: torch.Tensor) -> torch.Tensor:
        return infer_sliding_window(
            model=self.model,
            img_tensor=image,
            grid_size=self.input_size,
            device=self.device,
            overlap_ratio=self.overlap_ratio,
            pad_size=self.pad_size,
        )

    def _log_prediction(
        self,
        prefix: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if self.palette is None:
            warnings.warn("No palette defined in class_map; skipping prediction logging.")
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
            preds = torch.where(mask, preds, preds.new_full((), self.ignore_index))

        return preds, max_probs
