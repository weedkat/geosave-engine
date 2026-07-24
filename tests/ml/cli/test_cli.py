from __future__ import annotations

import pytest
from jsonargparse import Namespace

from geosave_engine.ml.cli import GeosaveCLI


def test_default_artifacts_use_tensorboard_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cli = object.__new__(GeosaveCLI)
    config = Namespace(
        model_name="forest-run",
        trainer=Namespace(callbacks=None, logger=None),
    )
    cli.subcommand = "fit"
    cli.config = Namespace(fit=config)

    cli.before_instantiate_classes()

    checkpoint = config.trainer.callbacks[0]
    assert checkpoint["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"
    assert "dirpath" not in checkpoint["init_args"]

    assert [logger["class_path"] for logger in config.trainer.logger] == [
        "lightning.pytorch.loggers.TensorBoardLogger",
    ]
    assert config.trainer.logger[0]["init_args"] == {
        "save_dir": "artifacts",
        "name": "forest-run",
        "log_graph": True,
    }


def test_mlflow_logger_requires_explicit_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "workspace-experiment")
    cli = object.__new__(GeosaveCLI)
    config = Namespace(
        model_name="forest-run",
        trainer=Namespace(callbacks=None, logger=None),
    )
    cli.subcommand = "fit"
    cli.config = Namespace(fit=config)

    cli.before_instantiate_classes()

    assert [logger["class_path"] for logger in config.trainer.logger] == [
        "lightning.pytorch.loggers.TensorBoardLogger",
        "lightning.pytorch.loggers.MLFlowLogger",
    ]
    assert config.trainer.logger[1]["init_args"]["experiment_name"] == (
        "workspace-experiment"
    )
    assert config.trainer.logger[1]["init_args"]["tracking_uri"] == (
        "http://mlflow:5000"
    )
