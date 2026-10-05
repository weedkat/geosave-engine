from __future__ import annotations

from pathlib import Path
import runpy

import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml
from torch.multiprocessing.spawn import ProcessRaisedException

from geosave_engine.ml.cli import GeosaveCLI
from geosave_engine.ml.segmentation import supervised
from geosave_engine.model.chain import chain_step


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(
        self, image: torch.Tensor, offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = image * self.factor
        return logits if offset is None else logits + offset


class MappedSegmentationModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(outputs=("logits",))
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.factor


class Samples(Dataset):
    def __init__(
        self,
        samples: list[tuple[dict[str, torch.Tensor], torch.Tensor, str, torch.Tensor]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str, torch.Tensor]:
        return self.samples[index]


@pytest.fixture
def stages() -> dict[str, dict[str, object]]:
    return {
        "model": {
            "class_path": f"{__name__}.SegmentationModel",
            "init_args": {"num_classes": 2},
        }
    }


def test_task_defaults(stages):
    task = supervised.Module(model_chain=stages, ignore_index=7)
    task.configure_model()
    assert task.num_classes == 2
    assert isinstance(task.criterion, nn.CrossEntropyLoss)
    assert task.criterion.ignore_index == 7
    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3


@pytest.mark.parametrize("hook", ["on_validation_epoch_start", "on_test_epoch_start"])
def test_distributed_scene_evaluation_fails_before_tiles_are_split(stages, hook):
    task = supervised.Module(model_chain=stages)
    task.trainer = Trainer(accelerator="cpu", devices=2, strategy="ddp", logger=False)
    assert task.trainer.world_size == 2
    with pytest.raises(ValueError, match="single device.*complete scene"):
        getattr(task, hook)()


@pytest.mark.slow
def test_two_process_lightning_rejects_incomplete_scene_evaluation(stages, tmp_path):
    task, loader, trainer, _ = _evaluation(
        stages, tmp_path, devices=2, strategy="ddp_fork"
    )
    with pytest.raises(ProcessRaisedException, match="single device.*complete scene"):
        trainer.validate(task, dataloaders=loader, verbose=False)


def test_the_module_takes_no_data_shaped_arguments(stages):
    for argument in ("in_channels", "input_size"):
        with pytest.raises(TypeError, match=argument):
            supervised.Module(model_chain=stages, **{argument: 2})


class UncountedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.factor


def test_the_last_stage_states_the_class_count() -> None:
    task = supervised.Module(
        model_chain={
            "model": {
                "class_path": f"{__name__}.SegmentationModel",
                "init_args": {"num_classes": 3},
            }
        },
        class_thresholds=[0.2, 0.4, 0.6],
    )

    task.configure_model()

    assert task.num_classes == 3
    assert task.class_thresholds.tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert len(task.val_metrics) > 0


def test_a_last_stage_stating_no_class_count_is_refused() -> None:
    task = supervised.Module(
        model_chain={"model": {"class_path": f"{__name__}.UncountedModel"}},
    )

    with pytest.raises(TypeError, match="UncountedModel.*num_classes"):
        task.configure_model()


def test_thresholds_must_number_the_classes(stages) -> None:
    task = supervised.Module(model_chain=stages, class_thresholds=[0.5])

    with pytest.raises(ValueError, match="must have 2 entries, got 1"):
        task.configure_model()


def test_task_uses_registered_training_builders(stages) -> None:
    task = supervised.Module(
        model_chain=stages,
        criterion={"name": "cross_entropy", "init_args": {"ignore_index": -1}},
        optimizer={"name": "sgd", "init_args": {"lr": 0.1}},
        lr_scheduler={
            "name": "reduce_on_plateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
        },
    )
    task.configure_model()

    configured = task.configure_optimizers()

    assert isinstance(task.criterion, nn.CrossEntropyLoss)
    assert task.criterion.ignore_index == -1
    assert isinstance(configured["optimizer"], torch.optim.SGD)
    assert configured["lr_scheduler"]["monitor"] == "val_loss"


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
    task = supervised.Module(
        model_chain=stages,
        ignore_index=7,
        criterion=criterion,
    )
    assert task.criterion.ignore_index == expected


class Encoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.frozen = nn.Parameter(torch.zeros(in_channels), requires_grad=False)

    @chain_step(outputs=("features",))
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.weight.reshape(1, -1, 1, 1)


class Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.ones(num_classes))

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.weight.reshape(1, -1, 1, 1)


def test_optimizer_exact_groups_and_remaining_trainable_parameters():
    task = supervised.Module(
        model_chain={
            "encoder": {"class_path": f"{__name__}.Encoder"},
            "head": {
                "class_path": f"{__name__}.Head",
                "init_args": {"num_classes": 2},
            },
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


def test_optimizer_keeps_direct_chain_parameters_in_default_group(stages):
    task = supervised.Module(
        model_chain=stages,
        optimizer={
            "class_path": "torch.optim.SGD",
            "init_args": {"lr": 0.1},
            "groups": {"model": {"lr": 0.01}},
        },
    )
    task.configure_model()
    bias = nn.Parameter(torch.tensor(1.0))
    task.model.register_parameter("bias", bias)
    task.model.register_parameter(
        "frozen", nn.Parameter(torch.tensor(1.0), requires_grad=False)
    )

    optimizer = task.configure_optimizers()

    assert len(optimizer.param_groups) == 2
    stage, remaining = optimizer.param_groups
    assert stage["params"] == [task.model.get_submodule("model").factor]
    assert len(remaining["params"]) == 1
    assert remaining["params"][0] is bias
    assert stage["lr"] == 0.01
    assert remaining["lr"] == 0.1
    bias.grad = torch.tensor(2.0)
    optimizer.step()
    torch.testing.assert_close(bias, torch.tensor(0.8))


@pytest.mark.parametrize("group", ["mod", "MODEL", "model.factor"])
def test_unknown_optimizer_groups_fail_before_optimizer_construction(stages, group):
    task = supervised.Module(
        model_chain=stages,
        optimizer={
            "class_path": "torch.optim.SGD",
            "groups": {group: {"lr": 0.1}},
            "init_args": {"invalid_argument": True},
        },
    )
    task.configure_model()
    with pytest.raises(ValueError, match="Unknown model groups"):
        task.configure_optimizers()


def test_scheduler_metadata_and_plateau_monitoring(stages):
    task = supervised.Module(
        model_chain=stages,
        lr_scheduler={
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 3,
            "strict": False,
            "scheduler_name": "rate",
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
        task = supervised.Module(
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
        task = supervised.Module(
            model_chain=stages,
            **{field: {"class_path": path, "init_args": {"wrong": True}}},
        )
        task.configure_model()
        task.configure_optimizers()
    assert error.value.__cause__ is None


def test_template_preserves_task_optimizer_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "geosave_engine.ml.cli.GeosaveCLI",
        lambda **kwargs: calls.append(kwargs),
    )
    template = Path(__file__).parents[4] / "src/geosave_engine/templates/common/main.py"
    runpy.run_path(str(template), run_name="__main__")
    assert calls[0]["auto_configure_optimizers"] is False


def test_forward_preserves_prepared_inputs_and_routes_context(
    stages: dict[str, dict[str, object]],
) -> None:
    task = supervised.Module(
        model_chain=stages,
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


def test_forward_rejects_mapped_model_results() -> None:
    task = supervised.Module(
        model_chain={
            "model": {
                "class_path": f"{__name__}.MappedSegmentationModel",
                "init_args": {"num_classes": 2},
            }
        },
    )
    task.configure_model()
    image = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]])

    native = task.model(image=image)
    assert isinstance(native, dict)
    with pytest.raises(TypeError, match="must return logits as a tensor"):
        task(image=image)


def test_steps_accept_model_inputs_and_target_tuples(
    stages: dict[str, dict[str, object]],
) -> None:
    task = supervised.Module(
        model_chain=stages,
    )
    task.configure_model()
    image = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]])
    target = torch.tensor([[[0, 1], [1, 0]]])

    loss = task.training_step(
        (
            {"image": image},
            target,
            ["a/tile-0"],
            torch.ones_like(target, dtype=torch.bool),
        ),
        0,
    )

    expected = torch.nn.functional.cross_entropy(image, target)
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    "batch",
    [
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)),
        {"image": torch.zeros(1, 2, 2, 2), "target": torch.zeros(1, 2, 2)},
    ],
)
def test_incompatible_batches_raise_native_python_errors(
    stages: dict[str, dict[str, object]], batch: object
) -> None:
    task = supervised.Module(
        model_chain=stages,
    )
    task.configure_model()

    with pytest.raises((AttributeError, TypeError, ValueError)):
        task.training_step(batch, 0)  # type: ignore[arg-type]


def test_training_and_checkpoint_reload_preserve_construction(
    stages: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    task = supervised.Module(
        model_chain=stages,
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
                    torch.tensor([[0, 1], [1, 0]]),
                    "scene-a/tile-0",
                    torch.ones_like(torch.tensor([[0, 1], [1, 0]]), dtype=torch.bool),
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
    restored = supervised.Module.load_from_checkpoint(checkpoint, weights_only=False)

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
    stages: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    cli = GeosaveCLI(
        supervised.Module,
        run=False,
        save_config_callback=None,
        seed_everything_default=False,
        auto_configure_optimizers=False,
        args={
            "model": {
                "model_chain": stages,
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
    assert isinstance(task, supervised.Module)
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


def prediction_context(row):
    from geosave_engine.model.encoder.prithvi import location_coords

    return {"offset": torch.zeros(1, 1, 1), "centre": location_coords(row)}


def test_lightning_predict_tiles_stitches_logits_on_source_grid(stages, tmp_path):
    from geosave_engine.geodata import stack
    from tiler import Merger
    from tests.ml.test_inputs import _samples
    from geosave_engine.model.spec import ModelSpec, Ref
    from tests.ml.test_inputs import _raster

    scene = _raster(6, 8)
    parents = {"scene-a": stack({"image": scene})}
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"image": {"variables": ["B04", "B08"]}},
            "inputs": {
                "image": Ref("image"),
            },
            "context": {
                "call": f"{__name__}.prediction_context",
                "kwargs": {"row": Ref("row")},
            },
        }
    )
    samples = _samples(parents, (4, 4), overlap=2, halo=True, spec=spec)
    centres = {tuple(samples[number][0]["centre"].tolist()) for number in range(4)}
    loader = DataLoader(
        samples, batch_size=3, sampler=list(reversed(range(len(samples))))
    )
    task = supervised.Module(
        model_chain=stages,
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
    assert len(centres) > 1
    merger = Merger(
        samples.layouts["scene-a"], logits=2, window="hann", save_visits=False
    )
    indices = []
    for logits, index in predictions:
        indices.extend(index)
        for sample_id, values in zip(index, logits.numpy(), strict=True):
            merger.add(int(samples.reference.loc[sample_id, "tile_id"]), values)
    assert indices == list(reversed(samples.reference.id.tolist()))
    output = merger.merge(extra_padding=samples.padding["scene-a"])
    torch.testing.assert_close(
        torch.from_numpy(output).float(),
        scene.gs.to_tensor(),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("shape", [(2, 2, 3, 5), (2, 3, 2, 3, 5)])
def test_predict_preserves_per_tile_context_and_indices(stages, shape):
    task = supervised.Module(
        model_chain=stages,
    )
    task.configure_model()
    image = torch.zeros(shape)
    offset = torch.tensor([3.0, 7.0]).reshape(2, *([1] * (len(shape) - 1)))
    indices = ["scene-a/tile-19", "scene-b/tile-2"]
    logits, returned = task.predict_step(
        ({"image": image, "offset": offset}, indices), 0
    )
    torch.testing.assert_close(logits, image + offset)
    assert returned is indices


def _evaluation(stages, tmp_path, **trainer_options):
    """Build a module, a validation loader over two 20x20 samples, and a trainer."""
    from tests.ml.segmentation.supervised.test_data import _manifest, _spec

    task = supervised.Module(model_chain=stages)
    task.configure_model()
    scored = []
    for metrics in (task.val_metrics, task.test_metrics):
        update = metrics.update

        def record(logits, target, update=update):
            scored.append((tuple(logits.shape), tuple(target.shape)))
            return update(logits, target)

        metrics.update = record
    dataset = supervised.Dataset(_manifest(tmp_path / "data"), _spec())
    loader = DataLoader(dataset, batch_size=5)
    trainer = Trainer(
        accelerator="cpu",
        devices=trainer_options.pop("devices", 1),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        **trainer_options,
    )
    return task, loader, trainer, scored


def test_validation_scores_each_raster_whole_and_once(stages, tmp_path):
    task, loader, trainer, scored = _evaluation(stages, tmp_path)

    validation = trainer.validate(task, dataloaders=loader, verbose=False)

    # 32 tiles of 8x8 overlap; what is scored is the two 20x20 rasters they rebuild.
    assert scored == [((1, 2, 20, 20), (1, 20, 20))] * 2
    # The second band leads the first by 1.0 everywhere, so class 1 is predicted:
    # wrong on the sample labelled 0, right on the one labelled 1.
    expected = (
        torch.log1p(torch.tensor(1.0).exp()) + torch.log1p(torch.tensor(-1.0).exp())
    ) / 2
    assert validation[0]["val_loss"] == pytest.approx(expected.item(), abs=1e-2)
    assert validation[0]["val_iou_macro"] == pytest.approx(0.25)


def test_test_scores_each_raster_whole_and_once(stages, tmp_path):
    task, loader, trainer, scored = _evaluation(stages, tmp_path)

    result = trainer.test(task, dataloaders=loader, verbose=False)

    assert scored == [((1, 2, 20, 20), (1, 20, 20))] * 2
    assert result[0]["test_iou_macro"] == pytest.approx(0.25)


def test_a_validation_run_cut_short_scores_no_partial_raster(stages, tmp_path):
    task, loader, trainer, scored = _evaluation(stages, tmp_path, limit_val_batches=2)

    with pytest.warns(UserWarning, match="before any raster was whole"):
        validation = trainer.validate(task, dataloaders=loader, verbose=False)

    assert scored == []
    assert "val_loss" not in validation[0]


def test_completed_parent_reads_original_classes_instead_of_blending_tile_targets(
    stages,
):
    from types import SimpleNamespace
    import numpy as np
    from geosave_engine.geodata import raster, stack
    from tests.ml.test_inputs import _samples
    from odc.geo.geobox import GeoBox

    grid = GeoBox.from_bbox((0, 0, 80, 60), "EPSG:32748", resolution=10)
    image = raster({"image": np.zeros((6, 8), "float32")}, grid)
    labels = np.where(np.indices((6, 8))[1] < 4, 2.0, 4.0).astype("float32")
    labels[0, 0] = np.nan
    label = raster({"label": labels}, grid, nodata=np.nan)
    parents = {"a": stack({"image": image, "label": label})}
    samples = _samples(parents, (4, 4), overlap=2, halo=True)
    reference = samples.reference
    dataset = SimpleNamespace(
        parents=parents,
        target="label",
        reference=reference,
        layouts=samples.layouts,
        padding=samples.padding,
        spec=SimpleNamespace(tiles=SimpleNamespace(window="hann")),
    )
    loader = SimpleNamespace(dataset=dataset)
    task = supervised.Module(model_chain=stages)
    task.configure_model()
    task.on_validation_epoch_start()
    ids = list(reversed(reference.id.tolist()))
    logits = torch.zeros(len(ids), 2, 4, 4)
    valid = torch.ones(len(ids), 4, 4, dtype=torch.bool)
    completed = task._merge(logits, ids, valid, loader, 0)
    assert len(completed) == 1
    _, target = completed[0]
    assert set(target.unique().tolist()) == {2, 4, task.ignore_index}
    assert target[0, 0, 0] == task.ignore_index
    assert task._raster_count == 1
    task.on_validation_epoch_start()
    assert task._evaluation == {}


def test_completed_parent_excludes_input_invalid_pixels_from_scoring(stages, tmp_path):
    task, loader, _, _ = _evaluation(stages, tmp_path)
    task.on_validation_epoch_start()
    dataset = loader.dataset
    ids = dataset.reference.id.tolist()
    logits = torch.ones(len(ids), 2, 8, 8)
    masks = torch.ones(len(ids), 8, 8, dtype=torch.bool)
    # Exclude one parent pixel in every overlapping contribution.
    for i, row in enumerate(dataset.reference.itertuples()):
        for y in range(8):
            for x in range(8):
                if (
                    row.parent_id == "s0"
                    and row.row_off + y == 4
                    and row.col_off + x == 5
                ):
                    masks[i, y, x] = False
    results = task._merge(logits, ids, masks, loader, 0)
    first_logits, first_target = results[0]
    assert first_target[0, 4, 5] == task.ignore_index
    assert first_logits.isfinite().all()
    assert first_target[0, 5, 5] == 0


@pytest.mark.parametrize("all_invalid", [False, True])
def test_native_evaluation_retains_nodata_and_finite_overlap(stages, all_invalid):
    from types import SimpleNamespace
    import numpy as np
    from geosave_engine.geodata import raster, stack
    from tests.ml.test_inputs import _samples, _raster

    image = _raster(6, 8)
    labels = raster({"label": np.zeros((6, 8), "float32")}, image.gs.geobox)
    parents = {"a": stack({"image": image, "label": labels})}
    samples = _samples(parents, (4, 4), overlap=2, halo=True)
    dataset = SimpleNamespace(
        parents=parents,
        target="label",
        reference=samples.reference,
        layouts=samples.layouts,
        padding=samples.padding,
        spec=SimpleNamespace(tiles=SimpleNamespace(window="hann")),
    )
    task = supervised.Module(model_chain=stages)
    task.configure_model()
    task.on_validation_epoch_start()
    ids = samples.reference.id.tolist()
    logits = torch.zeros(len(ids), 2, 4, 4)
    logits[0] = float("nan")  # a bad contribution must not poison neighbours
    masks = torch.ones(len(ids), 4, 4, dtype=torch.bool)
    for i, row in enumerate(samples.reference.itertuples()):
        yy, xx = np.indices((4, 4))
        masks[i] = torch.from_numpy(
            ~((yy + row.row_off == 4) & (xx + row.col_off == 5))
        )
    if all_invalid:
        masks[:] = False
    results = task._merge(logits, ids, masks, SimpleNamespace(dataset=dataset), 0)
    assert task._raster_count == 1
    if all_invalid:
        assert results == []
    else:
        values, target = results[0]
        assert target[0, 4, 5] == task.ignore_index
        assert target[0, 3, 3] == 0
        assert values.isfinite().all()
        torch.testing.assert_close(values, torch.zeros_like(values))
