from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch import LightningModule
from typer.testing import CliRunner

from geosave_engine.cli.commands import upload as upload_module
from geosave_engine.cli.errors import AbortedByUserError, WorkspaceError
from geosave_engine.cli.main import app

runner = CliRunner()


class _DummyModule(LightningModule):
    """Tiny real LightningModule — stands in for the heavy production models."""

    def __init__(self, size: int = 4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(size, size)


def _clear_modules_namespace_cache() -> None:
    """Drop cached modules.* entries so tests reimport from their own tmp_path.

    Different tests each write their own modules/lightning_module.py under a
    fresh tmp_path — without this, Python's sys.modules cache would return
    whichever test happened to import "modules.lightning_module" first.
    """
    for name in list(sys.modules):
        if name == "modules" or name.startswith("modules."):
            del sys.modules[name]


def _write_checkpoint(path: Path, module: LightningModule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": "2.6.1",
        },
        path,
    )


# ── _import_model_class ──────────────────────────────────────────────────


def test_import_model_class_resolves_installed_class() -> None:
    cls = upload_module._import_model_class("pathlib.Path")

    assert cls is Path


def test_import_model_class_raises_on_unknown_module() -> None:
    with pytest.raises(WorkspaceError, match="Could not import model class"):
        upload_module._import_model_class("not_a_real_module.Thing")


def test_import_model_class_raises_on_unknown_attribute() -> None:
    with pytest.raises(WorkspaceError, match="Could not import model class"):
        upload_module._import_model_class("pathlib.NotAClass")


def test_import_model_class_resolves_workspace_local_namespace_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modules_namespace_cache()
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    (modules_dir / "lightning_module.py").write_text(
        "class CustomTask:\n    pass\n"
    )

    # Without workspace root on sys.path, this namespace package doesn't resolve.
    with pytest.raises(WorkspaceError, match="Could not import model class"):
        upload_module._import_model_class("modules.lightning_module.CustomTask")

    # syspath_prepend auto-reverts — no cross-test sys.path pollution.
    monkeypatch.syspath_prepend(str(tmp_path))

    cls = upload_module._import_model_class("modules.lightning_module.CustomTask")
    assert cls.__name__ == "CustomTask"


# ── _load_model_from_checkpoint ──────────────────────────────────────────


def test_load_model_from_checkpoint_round_trips_real_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump({"model": {"class_path": f"{__name__}._DummyModule"}})
    )
    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    original = _DummyModule(size=4)
    _write_checkpoint(checkpoint_path, original)

    loaded = upload_module._load_model_from_checkpoint(config_path, checkpoint_path)

    assert isinstance(loaded, _DummyModule)
    assert torch.equal(loaded.linear.weight, original.linear.weight)


def test_load_model_from_checkpoint_requires_class_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n")
    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"irrelevant")

    with pytest.raises(WorkspaceError, match="missing model.class_path"):
        upload_module._load_model_from_checkpoint(config_path, checkpoint_path)


def test_load_model_from_checkpoint_wraps_corrupted_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump({"model": {"class_path": f"{__name__}._DummyModule"}})
    )
    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"not a real checkpoint")

    with pytest.raises(WorkspaceError, match="Could not load checkpoint"):
        upload_module._load_model_from_checkpoint(config_path, checkpoint_path)


# ── _require_tracking_uri / _require_experiment_name ─────────────────────


def test_require_tracking_uri_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    assert upload_module._require_tracking_uri() == "http://mlflow:5000"


def test_require_tracking_uri_prompts_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        upload_module.questionary,
        "text",
        lambda *_a, **_k: type("F", (), {"ask": lambda self: "http://prompted:5000"})(),
    )

    assert upload_module._require_tracking_uri() == "http://prompted:5000"


def test_require_tracking_uri_aborts_on_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        upload_module.questionary,
        "text",
        lambda *_a, **_k: type("F", (), {"ask": lambda self: None})(),
    )

    with pytest.raises(AbortedByUserError):
        upload_module._require_tracking_uri()


def test_require_experiment_name_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "my-experiment")

    assert upload_module._require_experiment_name("DynamicWorld") == "my-experiment"


def test_require_experiment_name_prompts_with_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    seen_defaults = []

    def _fake_text(_message, default=None):
        seen_defaults.append(default)
        return type("F", (), {"ask": lambda self: "picked-name"})()

    monkeypatch.setattr(upload_module.questionary, "text", _fake_text)

    assert upload_module._require_experiment_name("DynamicWorld") == "picked-name"
    assert seen_defaults == ["DynamicWorld"]


# ── log_model (real mlflow, local file store — no network) ───────────────


def test_log_model_registers_and_round_trips(tmp_path: Path) -> None:
    import mlflow

    model = _DummyModule(size=4)
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    (modules_dir / "data_pipeline.py").write_text("# marker file\n")
    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    _write_checkpoint(checkpoint_path, model)
    tracking_dir = tmp_path / "mlruns"

    model_info = upload_module.log_model(
        model=model,
        name="TestModel",
        checkpoint_path=checkpoint_path,
        modules_dir=modules_dir,
        tracking_uri=f"file://{tracking_dir}",
        experiment_name="TestModel",
    )

    assert str(model_info.registered_model_version) == "1"

    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    reloaded = mlflow.pytorch.load_model("models:/TestModel/1")
    assert torch.equal(reloaded.linear.weight, model.linear.weight)

    code_dirs = list(tracking_dir.rglob("code"))
    assert code_dirs, "code_paths bundle missing from logged artifact"
    assert "data_pipeline.py" in os.listdir(code_dirs[0] / "modules")


# ── upload() end-to-end through the real CLI entrypoint ──────────────────


def test_upload_command_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import toml

    with (tmp_path / "geosave.toml").open("w") as file:
        toml.dump(
            {
                "project_name": "demo",
                "project_task": "semantic_segmentation",
                "project_method": "supervised",
            },
            file,
        )

    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    (modules_dir / "lightning_module.py").write_text(
        "\n".join(
            [
                "import torch",
                "from lightning.pytorch import LightningModule",
                "",
                "",
                "class CustomTask(LightningModule):",
                "    def __init__(self, size: int = 4) -> None:",
                "        super().__init__()",
                "        self.save_hyperparameters()",
                "        self.linear = torch.nn.Linear(size, size)",
                "",
            ]
        )
    )

    run_dir = tmp_path / "artifacts" / "CustomRun" / "version_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.dump({"model": {"class_path": "modules.lightning_module.CustomTask"}})
    )

    # Build the checkpoint via the real (soon-to-be-importable) class so
    # state_dict keys match exactly what upload() will reconstruct.
    _clear_modules_namespace_cache()
    monkeypatch.syspath_prepend(str(tmp_path))

    custom_module = importlib.import_module("modules.lightning_module")
    original = custom_module.CustomTask(size=4)
    _write_checkpoint(run_dir / "checkpoints" / "last.ckpt", original)

    tracking_dir = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tracking_dir}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "CustomRun")

    result = runner.invoke(
        app,
        ["upload", str(tmp_path), "--artifact", "CustomRun/version_0"],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "models:/CustomRun/1"
