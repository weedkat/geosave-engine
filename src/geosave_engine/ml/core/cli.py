# geosave_engine/ml/cli.py
from __future__ import annotations
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import os

ARTIFACTS_ROOT = "artifacts"
DEFAULT_RUN_NAME = "run"


class GeosaveCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME,
                            help="Logger experiment / folder name.")

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

        run_name = getattr(cfg, "run_name", DEFAULT_RUN_NAME) or DEFAULT_RUN_NAME
        default_callbacks = [
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": f"{ARTIFACTS_ROOT}/model/{run_name}",
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "filename": "epoch={epoch:02d}-val_loss={val_loss:.4f}",
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
        cfg.trainer.logger = [
            {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": f"{ARTIFACTS_ROOT}/tensorboard",
                    "name": run_name,
                    "log_graph": True,
                },
            },
            {
                "class_path": "lightning.pytorch.loggers.MLFlowLogger",
                "init_args": {
                    "experiment_name": run_name,
                    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", f"file:./{ARTIFACTS_ROOT}/mlflow"),
                },
            },
            {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {
                    "save_dir": f"{ARTIFACTS_ROOT}/csv",
                    "name": run_name,
                },
            },
        ]

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