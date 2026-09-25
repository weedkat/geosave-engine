# geosave_engine/ml/cli/cli.py
from __future__ import annotations

import os

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

ARTIFACTS_ROOT = "artifacts"
DEFAULT_MODEL_NAME = "model"


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
        self._apply_default_loggers()

    def _apply_default_loggers(self) -> None:
        cfg = self._subcommand_config()
        if getattr(cfg.trainer, "logger", None) not in (None, True):
            return  # user supplied a logger config — respect it entirely

        model_name = (
            getattr(cfg, "model_name", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME
        )
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
                        # MLflow's own name for this run's display label.
                        "run_name": model_name,
                        "tracking_uri": tracking_uri,
                    },
                }
            )

        cfg.trainer.logger = loggers

    def _subcommand_config(self):
        """Return the running subcommand's config section, else the whole config."""
        sub = getattr(self, "subcommand", None)
        return self.config[sub] if sub and sub in self.config else self.config
