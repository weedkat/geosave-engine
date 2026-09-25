"""Generated segmentation workspaces compose library Lightning classes."""

import runpy
import subprocess
import sys

import pytest
import torch
import yaml

from geosave_engine.cli.core.workspace import create_workspace
from geosave_engine.ml.cli import GeosaveCLI
from geosave_engine.ml.data import SemanticSegmentationDataModule
from geosave_engine.ml.tasks import SemanticSegmentationTask
from geosave_engine.workflow.specs import ModelSpec, Ref
from geosave_engine.cli.core.templates import get_tasks


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    create_workspace(tmp_path, "semantic_segmentation", "supervised")
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_segmentation_configs_agree_on_model_inputs(workspace):
    spec = ModelSpec.load(workspace / "configs/model_spec.yaml")
    config = yaml.safe_load((workspace / "configs/model.yaml").read_text())
    source = spec.sources["sentinel_2_l2a"]
    assert source.collection == "sentinel-2-l2a"
    assert source.variables == ("B02", "B03", "B04", "B08")
    assert tuple(map(str, source.endpoints)) == (
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        "https://stac.dataspace.copernicus.eu/v1/",
    )
    assert source.require_crs
    model = SemanticSegmentationTask(**config["model"]["init_args"])
    assert model.in_channels == len(source.variables) == 4
    assert spec.inference["image"].call == Ref("image.gs.to_tensor")
    assert spec.inference["image"].kwargs == {"dtype": "float32"}
    assert model.num_classes == 2
    assert config["data"]["class_path"] == (
        "geosave_engine.ml.data.SemanticSegmentationDataModule"
    )
    data = SemanticSegmentationDataModule(**config["data"]["init_args"])
    assert data.input_size == model.input_size
    assert not (workspace / "modules" / "data.py").exists()
    assert not (workspace / "configs" / "augmentation.yaml").exists()


def test_model_config_uses_current_task_constructor(workspace):
    config = yaml.safe_load((workspace / "configs/model.yaml").read_text())
    model = SemanticSegmentationTask(**config["model"]["init_args"])
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.optimizer_spec["name"] == "adamw"
    assert model.scheduler_spec["name"] == "cosine_annealing"


def test_generated_entrypoint_keeps_task_optimizer_ownership(workspace, monkeypatch):
    options = {}

    def observe_cli(self, **kwargs):
        options.update(kwargs)

    # Stop at the CLI boundary: entry-point execution must not start training.
    monkeypatch.setattr(GeosaveCLI, "__init__", observe_cli)
    runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert options["auto_configure_optimizers"] is False
    assert options["subclass_mode_model"] is True
    assert options["subclass_mode_data"] is True


def test_generated_entrypoint_parses_library_lightning_pair(
    workspace, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "fit",
            "--config",
            str(workspace / "configs/model.yaml"),
            "--print_config",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert stopped.value.code == 0
    config = yaml.safe_load(capsys.readouterr().out)
    assert config["model"]["init_args"]["in_channels"] == 4
    assert config["data"]["class_path"] == (
        "geosave_engine.ml.data.SemanticSegmentationDataModule"
    )


@pytest.mark.slow
def test_custom_lightning_workspace_runs_one_train_and_validation_batch(tmp_path):
    assert "lightning" in get_tasks()["custom"]
    create_workspace(tmp_path, "custom", "lightning")
    script = """
from modules.data import CustomDataModule
from modules.task import CustomTask
from geosave_engine.ml.cli import GeosaveCLI

cli = GeosaveCLI(
    run=False,
    args=["--config", "configs/model.yaml"],
    save_config_callback=None,
    auto_configure_optimizers=False,
    subclass_mode_model=True,
    subclass_mode_data=True,
)
assert isinstance(cli.model, CustomTask)
assert isinstance(cli.datamodule, CustomDataModule)
cli.datamodule.setup("fit")
train_batch = next(iter(cli.datamodule.train_dataloader()))
validation_batch = next(iter(cli.datamodule.val_dataloader()))
assert cli.model.training_step(train_batch, 0).ndim == 0
assert cli.model.validation_step(validation_batch, 0).ndim == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
