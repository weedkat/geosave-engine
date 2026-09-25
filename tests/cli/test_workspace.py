"""Generated segmentation workspaces consume prepared LitData samples."""

from importlib import import_module
import runpy
import sys

from litdata import StreamingDataLoader, StreamingDataset
from litdata.streaming.cache import Cache
import pytest
import torch
import yaml

from geosave_engine.cli.core.workspace import create_workspace
from geosave_engine.ml.cli import GeosaveCLI
from geosave_engine.ml.tasks import SemanticSegmentationTask
from geosave_engine.workflow.spec import ModelSpec


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    create_workspace(tmp_path, "semantic_segmentation", "supervised")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in ("modules.data", "modules"):
        sys.modules.pop(name, None)


def test_segmentation_configs_agree_on_input_and_output(workspace):
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
    legend = spec.outputs["prediction"].legend
    assert legend.class_map == {0: "background", 1: "vegetation"}
    assert legend.color_map == {0: "#000000", 1: "#00ff00"}
    assert model.num_classes == len(legend.class_map) == 2
    overlay = yaml.safe_load((workspace / "configs/augmentation.yaml").read_text())
    config["data"]["init_args"].update(overlay["data"]["init_args"])
    module_path, _, name = config["data"]["class_path"].rpartition(".")
    data = getattr(import_module(module_path), name)(**config["data"]["init_args"])
    assert data.input_size == model.input_size
    image = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    target = torch.tensor([[0, 0, 1], [1, 1, 0]])
    inputs, labels = data.collate([(image, target), (image, target)])
    assert inputs["image"].shape == (2, 4, 2, 3)
    assert labels.shape == (2, 2, 3)


def test_model_config_uses_current_task_constructor(workspace):
    config = yaml.safe_load((workspace / "configs/model.yaml").read_text())
    model = SemanticSegmentationTask(**config["model"]["init_args"])
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.optimizer_spec["class_path"] == "torch.optim.AdamW"
    assert (
        model.scheduler_spec["class_path"]
        == "torch.optim.lr_scheduler.CosineAnnealingLR"
    )


@pytest.mark.parametrize("channel_mask", [False, True])
def test_generated_data_streams_paired_augmented_tuple_batches(workspace, channel_mask):
    data_module = import_module("modules.data")
    directory = workspace / "prepared"
    cache = Cache(str(directory), chunk_size=2)
    image = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) / 24
    target = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.int64)
    if channel_mask:
        target = target.unsqueeze(0)
    for index in range(2):
        cache[index] = (image, target)
    cache.done()
    cache.merge()
    data = data_module.SegmentationDataModule(
        train_path=str(directory),
        val_path=str(directory),
        input_size=(2, 3),
        batch_size=2,
        num_workers=0,
        augmentations=[{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}],
    )
    data.setup("fit")
    train = data.train_dataloader()
    assert isinstance(train, StreamingDataLoader)
    assert isinstance(train.dataset, StreamingDataset)
    batch = next(iter(train))
    assert isinstance(batch, tuple)
    inputs, labels = batch
    assert list(inputs) == ["image"]
    torch.testing.assert_close(inputs["image"], image.flip(-1).repeat(2, 1, 1, 1))
    torch.testing.assert_close(labels, torch.stack([target.flip(-1)] * 2))
    assert labels.dtype == torch.int64
    validation = data.val_dataloader()
    assert isinstance(validation, StreamingDataLoader)
    inputs, labels = next(iter(validation))
    torch.testing.assert_close(inputs["image"], image.repeat(2, 1, 1, 1))
    torch.testing.assert_close(labels, torch.stack([target] * 2))


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


def test_generated_entrypoint_parses_model_and_augmentation_configs(
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
            "--config",
            str(workspace / "configs/augmentation.yaml"),
            "--print_config",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert stopped.value.code == 0
    config = yaml.safe_load(capsys.readouterr().out)
    assert config["model"]["init_args"]["in_channels"] == 4
    assert config["data"]["class_path"] == "modules.data.SegmentationDataModule"
    assert config["data"]["init_args"]["augmentations"] == [
        {"name": "RandomHorizontalFlip", "init_args": {"p": 0.5}},
        {"name": "RandomVerticalFlip", "init_args": {"p": 0.5}},
    ]
