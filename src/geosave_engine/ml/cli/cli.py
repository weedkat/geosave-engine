# geosave_engine/ml/cli/cli.py
from __future__ import annotations

import os

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

ARTIFACTS_ROOT = "artifacts"
DEFAULT_MODEL_NAME = "model"
PREDICTION_WRITER_CLASS_PATH = "geosave_engine.ml.callbacks.PredictionWriter"


class GeosaveCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--model_name",
            type=str,
            default=DEFAULT_MODEL_NAME,
            help="Model identity — used as the artifacts/logger folder name, "
            "default MLflow experiment/run name, and default registered model name at upload time.",
        )

    def before_instantiate_classes(self) -> None:
        self._append_default_callback()
        self._apply_default_loggers()

    def _append_default_callback(self) -> None:
        cfg = self._subcommand_config()
        callbacks = getattr(cfg.trainer, "callbacks", None)

        if callbacks is False:
            return
        if callbacks in (None, True):
            callbacks = []
            cfg.trainer.callbacks = callbacks
        elif not isinstance(callbacks, list):
            callbacks = [callbacks]
            cfg.trainer.callbacks = callbacks

        default_callbacks = [
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "filename": "epoch={epoch:02d}-val_loss={val_loss:.4f}",
                    "save_last": True,
                },
            },
            {
                "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                "init_args": {"logging_interval": "epoch"},
            },
            {
                "class_path": "lightning.pytorch.callbacks.RichProgressBar",
                "init_args": {},
            },
        ]

        existing = [self._callback_class_path(cb) for cb in callbacks]
        for default in default_callbacks:
            if default["class_path"] not in existing:
                callbacks.append(default)

        self._fill_prediction_writer_model_name(cfg, callbacks)

    def _fill_prediction_writer_model_name(self, cfg, callbacks: list) -> None:
        """Back-fill a user-declared PredictionWriter's required model_name.

        The callback itself is never auto-added — the user opts in by
        listing it under trainer.callbacks (output_dir/input_keys are
        deployment-specific, no sensible default). Once it's there, its
        model_name comes from the same top-level `model_name:` key
        loggers/artifacts already use — one source of truth, not a
        second value to remember to set. An explicit `init_args.model_name`
        already present is left untouched.
        """
        model_name = getattr(cfg, "model_name", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME
        for cb in callbacks:
            if isinstance(cb, dict) and cb.get("class_path") == PREDICTION_WRITER_CLASS_PATH:
                cb.setdefault("init_args", {}).setdefault("model_name", model_name)

    def _apply_default_loggers(self) -> None:
        cfg = self._subcommand_config()
        if getattr(cfg.trainer, "logger", None) not in (None, True):
            return  # user supplied a logger config — respect it entirely

        model_name = getattr(cfg, "model_name", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME
        loggers = [
            {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": ARTIFACTS_ROOT,
                    "name": model_name,
                    "log_graph": True,
                },
            },
        ]

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            loggers.append(
                {
                    "class_path": "lightning.pytorch.loggers.MLFlowLogger",
                    "init_args": {
                        "experiment_name": os.getenv(
                            "MLFLOW_EXPERIMENT_NAME", model_name
                        ),
                        # MLflow's own kwarg name for "this run's display label" —
                        # bridged from our model_name, not a mismatch: this run
                        # IS (an attempt at) that model, until upload registers one.
                        "run_name": model_name,
                        "tracking_uri": tracking_uri,
                    },
                }
            )

        cfg.trainer.logger = loggers

    def _subcommand_config(self):
        """
        {
        "fit": {
            "model": {...},
            "trainer": {...},
            "model_name": "default"
        },
        "subcommand": "fit"
        }
        """
        sub = getattr(self, "subcommand", None)
        return self.config[sub] if sub and sub in self.config else self.config

    @staticmethod
    def _callback_class_path(cb) -> str | None:
        if isinstance(cb, dict):
            return cb.get("class_path")
        return getattr(cb, "class_path", None)
