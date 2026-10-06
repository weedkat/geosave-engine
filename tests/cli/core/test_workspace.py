"""Generated segmentation workspaces compose library Lightning classes."""

from geosave_engine.geodata.stac import asset, item
import runpy
import subprocess
import sys

import pytest
import torch
import yaml

from geosave_engine.cli.core.workspace import create_workspace
from geosave_engine.model.spec import ModelSpec
from geosave_engine.ml.cli import GeosaveCLI
from geosave_engine.ml.segmentation import supervised
from geosave_engine.cli.core.templates import get_workspaces


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    create_workspace(tmp_path, "segmentation")
    assert (tmp_path / "configs/train.yaml").is_file()
    assert not (tmp_path / "configs/model.yaml").exists()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_segmentation_configs_agree_on_model_inputs(workspace):
    spec = ModelSpec.load(workspace / "configs/model_spec.yaml")
    config = yaml.safe_load((workspace / "configs/train.yaml").read_text())
    requirement = spec.rasters["sentinel_2_l2a"]
    assert requirement.stac is not None
    assert requirement.stac.collection == "sentinel-2-l2a"
    assert requirement.variables == ("B02", "B03", "B04", "B08")
    assert tuple(map(str, requirement.stac.endpoints)) == (
        "planetary_computer",
        "cdse",
    )
    assert requirement.require_crs
    supervised.Module(**config["model"]["init_args"])
    encoder = config["model"]["init_args"]["model_chain"]["encoder"]["init_args"]
    assert encoder["in_channels"] == len(requirement.variables) == 4
    head = config["model"]["init_args"]["model_chain"]["head"]
    assert head == {
        "name": "segmentation",
        "init_args": {"classes": ["background", "target"]},
    }
    assert config["data"]["class_path"] == (
        "geosave_engine.ml.segmentation.supervised.DataModule"
    )
    data = supervised.DataModule(**config["data"]["init_args"])
    assert data.spec == spec
    assert spec.chips is not None
    assert spec.chips.shape == (encoder["input_size"],) * 2
    assert spec.pixel_inputs == ("image",)
    assert len(spec.transforms["image"][0].init_args["mean"]) == 4
    assert [step["name"] for step in data.augmentations] == [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
    ]
    assert not (workspace / "modules" / "data.py").exists()
    assert not (workspace / "configs" / "augmentation.yaml").exists()


def test_model_config_uses_current_task_constructor(workspace):
    config = yaml.safe_load((workspace / "configs/train.yaml").read_text())
    model = supervised.Module(**config["model"]["init_args"])
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
            str(workspace / "configs/train.yaml"),
            "--print_config",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert stopped.value.code == 0
    config = yaml.safe_load(capsys.readouterr().out)
    chain = config["model"]["init_args"]["model_chain"]
    assert chain["encoder"]["init_args"] == {"in_channels": 4, "input_size": 224}
    assert config["data"]["init_args"]["spec"] == "configs/model_spec.yaml"
    assert config["data"]["class_path"] == (
        "geosave_engine.ml.segmentation.supervised.DataModule"
    )


@pytest.mark.slow
def test_segmentation_workspace_trains_on_prepared_samples(workspace):
    from pathlib import Path

    from geosave_engine.geodata import GeoVector

    examples = Path(__file__).parents[3] / "examples/data/dw_imagery"
    # Each sample folder holds one raster per layer, named after it.
    entries = [
        item.from_assets(
            {layer.stem: asset.from_path(layer) for layer in sorted(path.iterdir())},
            id=path.name,
        )
        for path in sorted(examples.iterdir())
        if path.is_dir()
    ]
    samples = GeoVector.from_items(entries)
    for split in ("train", "val"):
        (workspace / f"data/{split}").mkdir(parents=True, exist_ok=True)
        samples.gs.to_geoparquet(
            workspace / f"data/{split}/manifest.parquet", overwrite=True
        )

    # The example rasters hold digital numbers, 10000 times the template's reflectance.
    spec_path = workspace / "configs/model_spec.yaml"
    text = spec_path.read_text()
    for reflectance, digital in (
        ("[0.1105, 0.1355, 0.1552, 0.2743]", "[1105.0, 1355.0, 1552.0, 2743.0]"),
        ("[0.1809, 0.1757, 0.1888, 0.1742]", "[1809.0, 1757.0, 1888.0, 1742.0]"),
    ):
        assert reflectance in text
        text = text.replace(reflectance, digital)
    spec_path.write_text(text)

    config = yaml.safe_load((workspace / "configs/train.yaml").read_text())
    chain = config["model"]["init_args"]["model_chain"]
    chain["encoder"]["init_args"]["pretrained"] = False
    # The example labels are Dynamic World classes, numbered up to 10.
    chain["head"]["init_args"]["classes"] = [f"class_{number}" for number in range(11)]
    config["data"]["init_args"].update(batch_size=4, num_workers=0)
    # A checkpoint of this encoder is over a gigabyte, so none is written here.
    config["trainer"]["callbacks"] = [
        callback
        for callback in config["trainer"]["callbacks"]
        if "ModelCheckpoint" not in callback["class_path"]
    ]
    config["trainer"].update(
        max_epochs=1,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )
    # Stage order is the chain's build order, so the keys must not be sorted.
    (workspace / "configs/test.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    cli = GeosaveCLI(
        auto_configure_optimizers=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
        args=["fit", "--config", str(workspace / "configs/test.yaml")],
    )

    metrics = cli.trainer.callback_metrics
    assert cli.trainer.global_step == 2
    assert torch.isfinite(metrics["train_loss"])
    # Three 510x510 samples, each rebuilt from 16 tiles and scored whole.
    assert len(cli.datamodule.val_dataset) == 48
    assert metrics["val_loss"] < 20
    assert 0.0 <= metrics["val_iou_macro"] <= 1.0


@pytest.mark.slow
def test_custom_lightning_workspace_runs_one_train_and_validation_batch(tmp_path):
    assert "custom_lightning" in get_workspaces()
    create_workspace(tmp_path, "custom_lightning")
    assert (tmp_path / "configs/train.yaml").is_file()
    assert not (tmp_path / "configs/model.yaml").exists()
    script = """
from modules.data import CustomDataModule
from modules.task import CustomTask
from geosave_engine.ml.cli import GeosaveCLI

cli = GeosaveCLI(
    run=False,
    args=["--config", "configs/train.yaml"],
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
