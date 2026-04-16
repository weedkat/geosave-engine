from lightning import LightningModule
from geosave_engine.core.class_resolver import (
    instantiate_from_config,
    instantiate_from_config_build,
)

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.threshold = threshold

        self.model_name = model_name
        self.model_config = model_config
        self.optim_config = optim_config
        self.loss_config = loss_config

        dense_model = model_config.get("dense_model")

        if dense_model is not None and isinstance(dense_model, dict):
            self.model = instantiate_from_config_build(dense_model)
        else:
            raise ValueError("Model configuration must include a 'dense_model' key with a valid model configuration dictionary.")

        loss = loss_config.get("supervised")
        if loss is not None and isinstance(loss, dict):
            self.loss = instantiate_from_config_build(loss)
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
        optimizer_configs = []

        for raw_optim_cfg in self.optim_config:
            if not isinstance(raw_optim_cfg, dict):
                raise ValueError("Each optimizer config entry must be a dictionary.")

            optim_cfg = dict(raw_optim_cfg)
            scheduler_cfg_raw = optim_cfg.pop("scheduler", None)

            optimizer = instantiate_from_config(optim_cfg, model=self.model)
            optimizer_item: dict = {"optimizer": optimizer}

            if scheduler_cfg_raw is not None:
                if not isinstance(scheduler_cfg_raw, dict):
                    raise ValueError("optimizer.scheduler must be a dictionary when provided.")

                scheduler_cfg = dict(scheduler_cfg_raw) # copy to avoid mutating input config
                lightning_scheduler_cfg = scheduler_cfg.pop("config", None)

                scheduler = instantiate_from_config(scheduler_cfg, optimizer=optimizer)

                if lightning_scheduler_cfg is None:
                    optimizer_item["lr_scheduler"] = scheduler
                else:
                    if not isinstance(lightning_scheduler_cfg, dict):
                        raise ValueError("optimizer.scheduler.config must be a dictionary.")

                    optimizer_item["lr_scheduler"] = {
                        "scheduler": scheduler,
                        **dict(lightning_scheduler_cfg),
                    }

            optimizer_configs.append(optimizer_item)

        return tuple(optimizer_configs)

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
        """Compute and return training outputs/loss for one batch."""
        raise NotImplementedError("training_step must be implemented by the task module.")

    def on_train_epoch_end(self):
        """Run at the end of each training epoch."""
        return super().on_train_epoch_end()

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
        """Compute validation outputs/loss for one batch."""
        raise NotImplementedError("validation_step must be implemented by the task module.")

    def on_validation_epoch_end(self):
        """Run at the end of each validation epoch."""
        return super().on_validation_epoch_end()

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
        """Compute test outputs/loss for one batch."""
        raise NotImplementedError("test_step must be implemented by the task module.")

    def on_test_epoch_end(self):
        """Run at the end of each test epoch."""
        return super().on_test_epoch_end()

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
        """Compute prediction outputs for one batch."""
        raise NotImplementedError("predict_step must be implemented by the task module.")

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