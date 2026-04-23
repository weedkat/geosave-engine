import torch
from lightning import LightningModule
from torchmetrics import MetricCollection

from geosave_engine.utils.ml.resolver import (
    instantiate_from_config_build,
    instantiate_optimizers_from_config,
)
from geosave_engine.ml.core.metrics import get_segmentation_metrics

class GeosaveLightningModule(LightningModule):
    """
    Base class for GeoSave lightning modules.
    Lightning modules are responsible for defining the neural network architecture and the training logic.
    They should implement the `training_step`, `validation_step`, and `test_step` methods to handle the training, validation, and testing loops.
    """

    def __init__(
        self,
        model_name: str,
        model_config: dict,
        optim_config: list[dict],
        loss_config: dict,
        threshold=0,
        threshold_callibration=True,
        log_image_every_n_steps: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.threshold = threshold
        self.log_image_every_n_steps = log_image_every_n_steps

        self.model_name = model_name
        self.model_config = model_config
        self.optim_config = optim_config
        self.loss_config = loss_config

        dense_model = model_config.get("dense_model")

        if dense_model is not None and isinstance(dense_model, dict):
            self.model: torch.nn.Module = instantiate_from_config_build(dense_model)  # type: ignore[assignment]
        else:
            raise ValueError("Model configuration must include a 'dense_model' key with a valid model configuration dictionary.")

        loss = loss_config.get("supervised")
        if loss is not None and isinstance(loss, dict):
            self.loss: torch.nn.Module = instantiate_from_config_build(loss)  # type: ignore[assignment]
        else:
            raise ValueError("Loss configuration must include a 'supervised' key with a valid loss configuration dictionary.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------
    def configure_model(self):
        """Create modules in a strategy/precision-aware context."""
        return super().configure_model()

    def configure_callbacks(self):
        """Return model-specific callbacks to merge with Trainer callbacks."""
        return []

    def configure_optimizers(self):
        """Choose optimizer and optional LR scheduler configuration."""
        return instantiate_optimizers_from_config(self.optim_config, model=self.model)

    def setup(self, stage: str | None = None) -> None:
        class_meta    = self.trainer.datamodule.class_meta  # type: ignore[union-attr]
        ignore_mask   = class_meta["ignore"].astype(bool)
        self.num_classes  = int((~ignore_mask).sum())
        self.ignore_index = int(class_meta[ignore_mask]["class_id"].iloc[0])
        class_names   = (
            class_meta[~ignore_mask]
            .sort_values("class_id")["class_name"]
            .tolist()
        )
        metrics = get_segmentation_metrics(self.num_classes, class_names, self.ignore_index)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics   = metrics.clone(prefix="val_")
        self.test_metrics  = metrics.clone(prefix="test_")

    # ---------------------------------------------------------------------
    # Common LightningModule utilities
    # ---------------------------------------------------------------------
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        weights_only=None,
        **kwargs,
    ):
        """Load a model from a checkpoint."""
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Fit lifecycle
    # ---------------------------------------------------------------------
    def on_fit_start(self):
        """Run at the very beginning of fit."""
        return super().on_fit_start()

    def on_fit_end(self):
        """Run at the very end of fit."""
        return super().on_fit_end()

    # ---------------------------------------------------------------------
    # Training lifecycle
    # ---------------------------------------------------------------------
    def on_train_start(self):
        """Run at the start of the training loop."""
        return super().on_train_start()

    def on_train_epoch_start(self):
        """Run at the beginning of each training epoch."""
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        logits = self(image)
        loss = self.loss(logits, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(logits, label)
        if self.global_step % self.log_image_every_n_steps == 0:
            self._log_segmentation("train", label, logits)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_train_end(self):
        """Run at the end of the training loop."""
        return super().on_train_end()

    # ---------------------------------------------------------------------
    # Validation lifecycle
    # ---------------------------------------------------------------------
    def on_validation_start(self):
        """Run at the start of validation."""
        return super().on_validation_start()

    def on_validation_epoch_start(self):
        """Run at the beginning of each validation epoch."""
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        logits = self(image)
        loss = self.loss(logits, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.update(logits, label)
        if batch_idx == 0:
            self._log_segmentation("val", label, logits)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_validation_end(self):
        """Run at the end of validation."""
        return super().on_validation_end()

    # ---------------------------------------------------------------------
    # Test lifecycle
    # ---------------------------------------------------------------------
    def on_test_start(self):
        """Run at the start of testing."""
        return super().on_test_start()

    def on_test_epoch_start(self):
        """Run at the beginning of each test epoch."""
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch["image"], batch["label"]
        logits = self(image)
        loss = self.loss(logits, label)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(logits, label)
        return loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def on_test_end(self):
        """Run at the end of testing."""
        return super().on_test_end()

    # ---------------------------------------------------------------------
    # Predict lifecycle
    # ---------------------------------------------------------------------
    def on_predict_start(self):
        """Run at the start of prediction."""
        return super().on_predict_start()

    def on_predict_epoch_start(self):
        """Run at the beginning of each prediction epoch."""
        return super().on_predict_epoch_start()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["image"]).argmax(dim=1)

    def on_predict_epoch_end(self):
        """Run at the end of each prediction epoch."""
        return super().on_predict_epoch_end()

    def on_predict_end(self):
        """Run at the end of prediction."""
        return super().on_predict_end()

    # ---------------------------------------------------------------------
    # Checkpoint hooks
    # ---------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        """Save additional state into the checkpoint dictionary."""
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Restore additional state from the checkpoint dictionary."""
        return super().on_load_checkpoint(checkpoint)

    # ---------------------------------------------------------------------
    # Non Lightning methods for model-specific logic
    # ---------------------------------------------------------------------

    def calibrate_thresholds(self):
        """Calibrate prediction thresholds based on validation outputs."""
        pass  # No-op, threshold calibration logic is handled in callbacks in this template

    def _log_segmentation(self, prefix: str, label: torch.Tensor, logits: torch.Tensor) -> None:
        if not hasattr(self.logger, "experiment"):
            return
        writer = self.logger.experiment
        step   = self.global_step
        scale  = float(self.num_classes - 1) or 1.0
        lbl    = label[0].float()
        pred   = logits[0].argmax(dim=0).float()
        writer.add_image(f"{prefix}/label",      (lbl  / scale).unsqueeze(0), step)  # type: ignore[attr-defined]
        writer.add_image(f"{prefix}/prediction", (pred / scale).unsqueeze(0), step)  # type: ignore[attr-defined]