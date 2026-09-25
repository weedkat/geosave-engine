from __future__ import annotations

from pathlib import Path
import runpy

import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from geosave_engine.ml.cli import GeosaveCLI
from geosave_engine.ml.models.contract import chain_step
from geosave_engine.ml.registry import StageSpec
from geosave_engine.ml.tasks import SemanticSegmentationTask


class SegmentationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: int | tuple[int, int],
        num_classes: int,
    ) -> None:
        super().__init__()
        assert in_channels == num_classes
        self.input_size = input_size
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(
        self, image: torch.Tensor, offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = image * self.factor
        return logits if offset is None else logits + offset


class Samples(Dataset):
    def __init__(self, samples: list[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return self.samples[index]


@pytest.fixture
def stages() -> dict[str, StageSpec]:
    return {"model": {"class_path": f"{__name__}.SegmentationModel"}}


def test_task_defaults_and_authoritative_dimensions(stages):
    stages["model"]["init_args"] = {
        "in_channels": 9,
        "num_classes": 8,
        "input_size": 17,
    }
    task = SemanticSegmentationTask(
        in_channels=2,
        num_classes=2,
        model_chain=stages,
        input_size=3,
        ignore_index=7,
    )
    task.configure_model()
    assert task.model.get_submodule("model").input_size == (3, 3)
    assert stages["model"]["init_args"] == {
        "in_channels": 9,
        "num_classes": 8,
        "input_size": 17,
    }
    assert isinstance(task.criterion, nn.CrossEntropyLoss)
    assert task.criterion.ignore_index == 7
    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3


@pytest.mark.parametrize(
    "criterion, expected",
    [
        ({"class_path": "torch.nn.CrossEntropyLoss"}, 7),
        (
            {
                "class_path": "torch.nn.CrossEntropyLoss",
                "init_args": {"ignore_index": -1},
            },
            -1,
        ),
    ],
)
def test_criterion_ignore_index(stages, criterion, expected):
    task = SemanticSegmentationTask(
        in_channels=2,
        num_classes=2,
        model_chain=stages,
        ignore_index=7,
        criterion=criterion,
    )
    assert task.criterion.ignore_index == expected


class Encoder(nn.Module):
    def __init__(self, in_channels, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.frozen = nn.Parameter(torch.zeros(in_channels), requires_grad=False)

    @chain_step(outputs=("features",))
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.weight.reshape(1, -1, 1, 1)


class Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_classes))

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.weight.reshape(1, -1, 1, 1)


def test_optimizer_exact_groups_and_remaining_trainable_parameters():
    task = SemanticSegmentationTask(
        in_channels=2,
        num_classes=2,
        model_chain={
            "encoder": {"class_path": f"{__name__}.Encoder"},
            "head": {"class_path": f"{__name__}.Head"},
        },
        optimizer={
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 1e-3, "weight_decay": 1e-2},
            "groups": {"head": {"lr": 1e-4, "weight_decay": 0.0}},
        },
    )
    task.configure_model()
    optimizer = task.configure_optimizers()
    assert len(optimizer.param_groups) == 2
    head, remaining = optimizer.param_groups
    assert head["params"] == [task.model.get_submodule("head").weight]
    assert remaining["params"] == [task.model.get_submodule("encoder").weight]
    assert (head["lr"], head["weight_decay"]) == (1e-4, 0.0)
    assert (remaining["lr"], remaining["weight_decay"]) == (1e-3, 1e-2)


@pytest.mark.parametrize("group", ["mod", "MODEL", "model.factor"])
def test_unknown_optimizer_groups_fail_before_optimizer_construction(stages, group):
    task = SemanticSegmentationTask(
        in_channels=2,
        num_classes=2,
        model_chain=stages,
        optimizer={
            "class_path": "torch.optim.SGD",
            "groups": {group: {"lr": 0.1}},
            "init_args": {"invalid_argument": True},
        },
    )
    task.configure_model()
    with pytest.raises(ValueError, match="Unknown model-chain optimizer groups"):
        task.configure_optimizers()


def test_scheduler_metadata_and_plateau_monitoring(stages):
    task = SemanticSegmentationTask(
        in_channels=2,
        num_classes=2,
        model_chain=stages,
        lr_scheduler={
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 3,
            "strict": False,
            "name": "rate",
        },
    )
    task.configure_model()
    configured = task.configure_optimizers()
    scheduler = configured["lr_scheduler"].pop("scheduler")
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.optimizer is configured["optimizer"]
    assert scheduler.patience == 2
    assert configured["lr_scheduler"] == {
        "monitor": "val_loss",
        "interval": "epoch",
        "frequency": 3,
        "strict": False,
        "name": "rate",
    }


@pytest.mark.parametrize(
    "field, path",
    [
        ("criterion", "torch.optim.AdamW"),
        ("optimizer", "torch.nn.CrossEntropyLoss"),
        ("lr_scheduler", "torch.nn.CrossEntropyLoss"),
    ],
)
def test_training_specs_require_expected_torch_subclass(stages, field, path):
    with pytest.raises(TypeError, match="subclass"):
        task = SemanticSegmentationTask(
            in_channels=2,
            num_classes=2,
            model_chain=stages,
            **{field: {"class_path": path}},
        )
        task.configure_model()
        task.configure_optimizers()


@pytest.mark.parametrize(
    "field, path",
    [
        ("criterion", "torch.nn.CrossEntropyLoss"),
        ("optimizer", "torch.optim.AdamW"),
        ("lr_scheduler", "torch.optim.lr_scheduler.StepLR"),
    ],
)
def test_training_constructor_errors_remain_native(stages, field, path):
    with pytest.raises(TypeError, match="unexpected keyword argument 'wrong'") as error:
        task = SemanticSegmentationTask(
            in_channels=2,
            num_classes=2,
            model_chain=stages,
            **{field: {"class_path": path, "init_args": {"wrong": True}}},
        )
        task.configure_model()
        task.configure_optimizers()
    assert error.value.__cause__ is None


def test_template_preserves_task_optimizer_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "geosave_engine.ml.cli.GeosaveCLI", lambda **kwargs: calls.append(kwargs)
    )
    template = Path(__file__).parents[3] / "src/geosave_engine/templates/common/main.py"
    runpy.run_path(str(template), run_name="__main__")
    assert calls[0]["auto_configure_optimizers"] is False


def test_forward_preserves_prepared_inputs_and_routes_context(
    stages: dict[str, StageSpec],
) -> None:
    task = SemanticSegmentationTask(
        model_chain=stages,
        in_channels=2,
        num_classes=2,
        input_size=2,
    )
    task.configure_model()
    image = torch.tensor([[[[0.0, 1000.0], [2000.0, 3000.0]]] * 2], dtype=torch.float64)
    original = image.clone()
    offset = torch.tensor(4.0)

    actual = task(image=image, offset=offset)

    torch.testing.assert_close(actual, image + offset)
    torch.testing.assert_close(image, original)
    assert actual.dtype == torch.float64
    assert not hasattr(task, "preprocessor")
    assert not hasattr(task, "preprocess")


def test_steps_accept_model_inputs_and_target_tuples(stages: dict[str, StageSpec]) -> None:
    task = SemanticSegmentationTask(
        model_chain=stages,
        in_channels=2,
        num_classes=2,
        input_size=2,
    )
    task.configure_model()
    task.setup("fit")
    image = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]])
    target = torch.tensor([[[[0, 1], [1, 0]]]])

    loss = task.training_step(({"image": image}, target), 0)

    expected = torch.nn.functional.cross_entropy(image, target.squeeze(1))
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    "batch",
    [
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)),
        {"image": torch.zeros(1, 2, 2, 2), "target": torch.zeros(1, 2, 2)},
    ],
)
def test_incompatible_batches_raise_native_python_errors(
    stages: dict[str, StageSpec], batch: object
) -> None:
    task = SemanticSegmentationTask(
        model_chain=stages,
        in_channels=2,
        num_classes=2,
        input_size=2,
    )
    task.configure_model()
    task.setup("fit")

    with pytest.raises((AttributeError, TypeError, ValueError)):
        task.training_step(batch, 0)  # type: ignore[arg-type]


def test_training_and_checkpoint_reload_preserve_construction(
    stages: dict[str, StageSpec], tmp_path: Path
) -> None:
    task = SemanticSegmentationTask(
        model_chain=stages,
        in_channels=2,
        num_classes=2,
        input_size=2,
        optimizer={"class_path": "torch.optim.SGD", "init_args": {"lr": 0.1}},
        criterion={
            "class_path": "torch.nn.CrossEntropyLoss",
            "init_args": {"ignore_index": -1},
        },
        lr_scheduler={
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": 2},
            "interval": "step",
        },
    )
    image = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    loader = DataLoader(
        Samples(
            [
                (
                    {"image": image, "offset": torch.tensor(0.0)},
                    torch.tensor([[[0, 1], [1, 0]]]),
                )
            ]
        ),
        batch_size=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(task, train_dataloaders=loader)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = SemanticSegmentationTask.load_from_checkpoint(
        checkpoint, weights_only=False
    )

    assert trainer.global_step == 1
    model = task.model.get_submodule("model")
    assert isinstance(model, SegmentationModel)
    assert model.factor.item() > 1.0
    torch.testing.assert_close(
        restored(image=image.unsqueeze(0), offset=torch.tensor(0.0)),
        task(image=image.unsqueeze(0), offset=torch.tensor(0.0)),
    )
    assert restored.hparams["model_chain"] == stages
    assert restored.hparams == task.hparams
    configured = restored.configure_optimizers()
    assert isinstance(configured["optimizer"], torch.optim.SGD)
    assert configured["lr_scheduler"]["interval"] == "step"
    assert restored.criterion.ignore_index == -1


def test_lightning_cli_parses_nested_construction_specs(
    stages: dict[str, StageSpec], tmp_path: Path
) -> None:
    cli = GeosaveCLI(
        SemanticSegmentationTask,
        run=False,
        save_config_callback=None,
        seed_everything_default=False,
        auto_configure_optimizers=False,
        args={
            "model": {
                "model_chain": stages,
                "in_channels": 2,
                "num_classes": 2,
                "criterion": {
                    "class_path": "torch.nn.CrossEntropyLoss",
                    "init_args": {"ignore_index": -1},
                },
                "optimizer": {
                    "class_path": "torch.optim.SGD",
                    "init_args": {"lr": 0.1},
                    "groups": {"model": {"lr": 0.01}},
                },
                "lr_scheduler": {
                    "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "logger": False,
                "enable_progress_bar": False,
                "default_root_dir": str(tmp_path),
            },
        },
    )
    task = cli.model
    assert isinstance(task, SemanticSegmentationTask)
    task.configure_model()
    configured = task.configure_optimizers()
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert task.criterion.ignore_index == -1
    assert configured["lr_scheduler"]["monitor"] == "val_loss"
    config = yaml.safe_load(cli.parser.dump(cli.config))
    assert isinstance(config, dict)
    assert config["model"]["optimizer"]["init_args"] == {"lr": 0.1}


def test_lightning_predict_tiles_stitches_logits_on_source_grid(stages, tmp_path):
    from geosave_engine.geodata.datasets import TileDataset
    from geosave_engine.geodata.transform.tiling import Tiles
    from tests.geodata.datasets.test_tiles import _raster

    scene = _raster(6, 8)
    tiles = Tiles([scene], (4, 4), overlap=2)
    seen_centres = []

    def context(tile):
        seen_centres.append(tile.gs.anchor.geographic_centroid)
        return {"offset": torch.zeros(1, 1, 1)}

    samples = TileDataset(tiles, model_context=context)
    loader = DataLoader(
        samples, batch_size=3, sampler=list(reversed(range(len(tiles))))
    )
    task = SemanticSegmentationTask(
        model_chain=stages,
        in_channels=2,
        num_classes=2,
        input_size=4,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    predictions = trainer.predict(task, dataloaders=loader)
    assert predictions is not None
    assert len(set(seen_centres)) > 1
    merger = tiles.merger(window="hann")
    indices = []
    for logits, index in predictions:
        indices.extend(index.tolist())
        merger.add(dict(zip(index.tolist(), logits.numpy(), strict=True)))
    assert indices == list(reversed(range(len(tiles))))
    output = merger.merge()[0]
    assert output.odc.geobox == scene.odc.geobox
    torch.testing.assert_close(
        torch.from_numpy(output.values).float(),
        scene.gs.to_tensor(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert merger.pending == {}


@pytest.mark.parametrize("shape", [(2, 2, 3, 5), (2, 3, 2, 3, 5)])
def test_predict_preserves_per_tile_context_and_indices(stages, shape):
    task = SemanticSegmentationTask(
        model_chain=stages,
        in_channels=2,
        num_classes=2,
        input_size=2,
    )
    task.configure_model()
    image = torch.zeros(shape)
    offset = torch.tensor([3.0, 7.0]).reshape(2, *([1] * (len(shape) - 1)))
    indices = torch.tensor([19, 2])
    logits, returned = task.predict_step(({"image": image, "offset": offset}, indices), 0)
    torch.testing.assert_close(logits, image + offset)
    assert returned is indices


def test_validation_and_test_evaluate_prepared_tiles(stages, tmp_path):
    task = SemanticSegmentationTask(
        model_chain=stages,
        in_channels=2,
        num_classes=2,
        input_size=2,
    )
    image = torch.tensor([[[4.0, 0.0, 4.0]], [[0.0, 4.0, 0.0]]])
    loader = DataLoader(
        Samples(
            [
                (
                    {"image": image, "offset": torch.tensor(0.0)},
                    torch.tensor([[[0, 1, 0]]]),
                )
            ]
        ),
        batch_size=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    validation = trainer.validate(task, dataloaders=loader, verbose=False)
    expected = torch.nn.functional.cross_entropy(
        image.unsqueeze(0),
        torch.tensor([[[0, 1, 0]]]),
    )
    assert validation[0]["val_loss"] == pytest.approx(expected.item())
    result = trainer.test(task, dataloaders=loader, verbose=False)
    assert result[0]["test_iou_macro"] == pytest.approx(1.0)
