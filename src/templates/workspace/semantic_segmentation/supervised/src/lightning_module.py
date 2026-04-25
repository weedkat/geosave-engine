from __future__ import annotations

from typing import cast

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger

from geosave_engine.ml.callbacks.calibration import CalibrationCallback
from geosave_engine.ml.core.metrics import get_segmentation_metrics
from geosave_engine.utils.geodata.manifest import TrainingMeta
from geosave_engine.utils.ml.resolver import (
    instantiate_from_config_build,
    instantiate_optimizers_from_config,
)


class GeosaveLightningModule(LightningModule):
    """Semantic-segmentation Lightning module driven by manifest metadata.

    ``in_channels``, ``num_classes``, and ``ignore_index`` are pulled from
    ``self.trainer.datamodule`` (via ``training_meta`` + ``in_channels``) at
    ``configure_model`` time, so the YAML config does not need to repeat them.
    """

    def __init__(
        self,
        model_config: dict,
        optim_config: list[dict],
        loss_config: dict,
        threshold_calibration: bool = False,
        log_image_every_n_steps: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optim_config = optim_config
        self.loss_config = loss_config
        self.threshold_calibration = threshold_calibration
        self.log_image_every_n_steps = log_image_every_n_steps

        self.calibrating: bool = False

        if "dense_model" not in model_config or not isinstance(model_config["dense_model"], dict):
            raise ValueError("model_config must include a 'dense_model' dict.")
        if "supervised" not in loss_config or not isinstance(loss_config["supervised"], dict):
            raise ValueError("loss_config must include a 'supervised' dict.")

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------
    def configure_model(self) -> None:
        """Build dense model + loss using manifest-derived shape from the datamodule."""
        if hasattr(self, "model"):
            return  # idempotent — Lightning may call this more than once

        meta = self._training_meta()
        in_channels = int(self._datamodule().in_channels)

        dense_cfg = dict(self.model_config["dense_model"])
        dense_cfg["in_channels"] = in_channels
        dense_cfg["classes"] = meta.num_classes
        self.model: torch.nn.Module = instantiate_from_config_build(dense_cfg)

        loss_cfg = dict(self.loss_config["supervised"])
        loss_cfg.setdefault("ignore_index", meta.ignore_index)
        self.loss: torch.nn.Module = instantiate_from_config_build(loss_cfg)

        self.num_classes = meta.num_classes
        self.ignore_index = meta.ignore_index
        self.register_buffer(
            "class_thresholds",
            torch.full((self.num_classes,), 0.5),
        )

    def configure_callbacks(self):
        if not self.threshold_calibration:
            return []
        return [CalibrationCallback()]

    def configure_optimizers(self):
        return instantiate_optimizers_from_config(self.optim_config, model=self.model)

    def setup(self, stage: str | None = None) -> None:
        from copy import deepcopy
        meta = self._training_meta()
        metrics = get_segmentation_metrics(
            num_classes=meta.num_classes,
            class_names=meta.class_names,
            ignore_index=meta.ignore_index,
        )
        self.train_scalar = metrics.scalar.clone(prefix="train_")
        self.val_scalar = metrics.scalar.clone(prefix="val_")
        self.test_scalar = metrics.scalar.clone(prefix="test_")
        self.train_per_class = deepcopy(metrics.per_class)
        self.val_per_class = deepcopy(metrics.per_class)
        self.test_per_class = deepcopy(metrics.per_class)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Training / validation / test
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        logits = self(image)
        loss = self.loss(logits, label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_scalar.update(logits, label)
        self.train_per_class.update(logits, label)

        if self.global_step % self.log_image_every_n_steps == 0:
            self._log_segmentation("train", label, logits)

        return loss

    def on_train_epoch_end(self):
        train_metrics = {
            **self.train_scalar.compute(),
            **self._prefixed(self.train_per_class.compute(), "train_"),
        }

        self.log_dict(train_metrics, on_step=False, on_epoch=True)
        self.train_scalar.reset()
        self.train_per_class.reset()
        # Lightning fires `_logger_connector.on_epoch_end()` AFTER this hook,
        # so val metrics from the just-finished epoch are already in
        # `callback_metrics` while train metrics aren't yet — pass the freshly
        # computed train dict to the monitor so the rendered table has both.
        self._render_epoch_summary(train_metrics)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        logits = self(image)
        loss = self.loss(logits, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_scalar.update(logits, label)
        self.val_per_class.update(logits, label)
        
        if batch_idx == 0:
            self._log_segmentation("val", label, logits)

        if self.calibrating:
            probs = logits.softmax(dim=1)
            max_probs, preds = probs.max(dim=1)
            return preds, max_probs, label

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_scalar.compute(), on_step=False, on_epoch=True)
        self.log_dict(self._prefixed(self.val_per_class.compute(), "val_"), on_step=False, on_epoch=True)
        self.val_scalar.reset()
        self.val_per_class.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        logits = self(image)
        loss = self.loss(logits, label)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_scalar.update(logits, label)
        self.test_per_class.update(logits, label)
        return loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_scalar.compute(), on_step=False, on_epoch=True)
        self.log_dict(self._prefixed(self.test_per_class.compute(), "test_"), on_step=False, on_epoch=True)
        self.test_scalar.reset()
        self.test_per_class.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["image"])
        return self.postprocess(logits)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _prefixed(d: dict, prefix: str) -> dict:
        return {f"{prefix}{k}": v for k, v in d.items()}

    def _render_epoch_summary(self, fresh_train_metrics: dict) -> None:
        """Push the epoch summary panel through ``LiveTrainingMonitor`` if attached.

        ``fresh_train_metrics`` is the just-computed train dict; val metrics
        from the same epoch are read from ``trainer.callback_metrics`` (already
        materialised by the val loop's ``logger_connector.on_epoch_end``).
        """
        from geosave_engine.ml.callbacks.training_monitor import LiveTrainingMonitor
        if self.trainer is None or self.trainer.sanity_checking:
            return
        monitor = next(
            (cb for cb in self.trainer.callbacks if isinstance(cb, LiveTrainingMonitor)),
            None,
        )
        if monitor is None:
            return
        merged = {**self.trainer.callback_metrics, **fresh_train_metrics}
        monitor.render_epoch_summary(self.trainer, merged)

    def _datamodule(self):
        dm = getattr(self.trainer, "datamodule", None) if self._trainer is not None else None
        if dm is None:
            raise RuntimeError(
                "GeosaveLightningModule requires a LightningDataModule with "
                "`training_meta` and `in_channels` attributes."
            )
        return dm

    def _training_meta(self) -> TrainingMeta:
        meta = getattr(self._datamodule(), "training_meta", None)
        if not isinstance(meta, TrainingMeta):
            raise RuntimeError(
                "datamodule.training_meta is missing or not a TrainingMeta instance."
            )
        return meta

    def _log_segmentation(self, prefix: str, labels: torch.Tensor, logits: torch.Tensor) -> None:
        tb_logger = next(
            (lg for lg in self.loggers if isinstance(lg, TensorBoardLogger)), None
        )
        if tb_logger is None:
            return
        writer = tb_logger.experiment
        step = self.global_step
        scale = float(self.num_classes - 1) or 1.0
        label = labels[0].float()
        pred = logits[0].argmax(dim=0).float()
        writer.add_image(f"{prefix}/label", (label / scale).unsqueeze(0), step)
        writer.add_image(f"{prefix}/prediction", (pred / scale).unsqueeze(0), step)

    def postprocess(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = logits.softmax(dim=1)
        max_probs, preds = probs.max(dim=1)

        if not self.calibrating:
            class_thresholds = cast(torch.Tensor, self.class_thresholds)
            pixel_thresholds = torch.index_select(class_thresholds, 0, preds.reshape(-1)).view_as(preds)
            preds = torch.where(
                max_probs >= pixel_thresholds,
                preds,
                preds.new_full((), self.ignore_index),
            )

        return preds, max_probs
