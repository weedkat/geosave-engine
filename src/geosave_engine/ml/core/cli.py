# geosave_engine/ml/cli.py
from __future__ import annotations

import os

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

ARTIFACTS_ROOT = "artifacts"
DEFAULT_RUN_NAME = "run"


class GeosaveCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--run_name",
            type=str,
            default=DEFAULT_RUN_NAME,
            help="Logger experiment / folder name.",
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

    def _apply_default_loggers(self) -> None:
        cfg = self._subcommand_config()
        if getattr(cfg.trainer, "logger", None) not in (None, True):
            return  # user supplied a logger config — respect it entirely

        run_name = getattr(cfg, "run_name", DEFAULT_RUN_NAME) or DEFAULT_RUN_NAME
        loggers = [
            {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": ARTIFACTS_ROOT,
                    "name": run_name,
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
                            "MLFLOW_EXPERIMENT_NAME", run_name
                        ),
                        "run_name": run_name,
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
            "run_name": "default"
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
