from __future__ import annotations

from pathlib import Path

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
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(self, image: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return image * self.factor + offset


class Samples(Dataset):
    def __init__(self, samples: list[dict[str, dict[str, torch.Tensor]]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        return self.samples[index]


@pytest.fixture
def stages() -> dict[str, StageSpec]:
    return {"model": {"class_path": f"{__name__}.SegmentationModel"}}


def test_forward_preserves_prepared_inputs_and_routes_context(
    stages: dict[str, StageSpec],
) -> None:
    task = SemanticSegmentationTask(
        stages=stages,
        class_map={0: "water", 1: "trees"},
        band_map={0: "B04", 1: "B08"},
        input_size=2,
    )
    task.configure_model()
    image = torch.tensor([[[[0.0, 1000.0], [2000.0, 3000.0]]] * 2], dtype=torch.float64)
    original = image.clone()
    offset = torch.tensor(4.0)

    actual = task(image, offset=offset)

    torch.testing.assert_close(actual, image + offset)
    torch.testing.assert_close(image, original)
    assert actual.dtype == torch.float64
    assert not hasattr(task, "preprocessor")
    assert not hasattr(task, "preprocess")


def test_training_and_checkpoint_reload_preserve_construction(
    stages: dict[str, StageSpec], tmp_path: Path
) -> None:
    task = SemanticSegmentationTask(
        stages=stages,
        class_map={0: "water", 1: "trees"},
        band_map={0: "B04", 1: "B08"},
        input_size=2,
        optimizer={"class_path": "torch.optim.SGD", "init_args": {"lr": 0.1}},
        scheduler={"name": "CosineAnnealingLR", "init_args": {"T_max": 2}},
    )
    image = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    loader = DataLoader(
        Samples(
            [
                {
                    "layers": {
                        "image": image,
                        "label": torch.tensor([[[0, 1], [1, 0]]]),
                    },
                    "model_context": {"offset": torch.tensor(0.0)},
                }
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
        restored(image.unsqueeze(0), offset=torch.tensor(0.0)),
        task(image.unsqueeze(0), offset=torch.tensor(0.0)),
    )
    assert restored.hparams["stages"] == stages


def test_lightning_cli_parses_nested_construction_specs(
    stages: dict[str, StageSpec], tmp_path: Path
) -> None:
    cli = GeosaveCLI(
        SemanticSegmentationTask,
        run=False,
        save_config_callback=None,
        seed_everything_default=False,
        args={
            "model": {
                "stages": stages,
                "class_map": {0: "water", 1: "trees"},
                "band_map": {0: "B04", 1: "B08"},
                "optimizer": {
                    "class_path": "torch.optim.SGD",
                    "init_args": {"lr": 0.1},
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
    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.1
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
        stages=stages,
        class_map={0: "water", 1: "trees"},
        band_map={0: "B04", 1: "B08"},
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
    for batch in predictions:
        assert isinstance(batch, dict)
        assert set(batch) == {"logits", "index"}
        indices.extend(batch["index"].tolist())
        merger.add(
            dict(zip(batch["index"].tolist(), batch["logits"].numpy(), strict=True))
        )
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
        stages=stages,
        class_map={0: "water", 1: "trees"},
        band_map={0: "B04", 1: "B08"},
        input_size=2,
    )
    task.configure_model()
    image = torch.zeros(shape)
    offset = torch.tensor([3.0, 7.0]).reshape(2, *([1] * (len(shape) - 1)))
    indices = torch.tensor([19, 2])
    result = task.predict_step(
        {
            "image": image,
            "index": indices,
            "model_context": {"offset": offset},
        },
        0,
    )
    torch.testing.assert_close(result["logits"], image + offset)
    assert result["index"] is indices


def test_validation_and_test_evaluate_prepared_tiles(stages, tmp_path):
    task = SemanticSegmentationTask(
        stages=stages,
        class_map={0: "water", 1: "trees"},
        band_map={0: "B04", 1: "B08"},
        input_size=2,
    )
    image = torch.tensor([[[4.0, 0.0, 4.0]], [[0.0, 4.0, 0.0]]])
    loader = DataLoader(
        Samples(
            [
                {
                    "layers": {"image": image, "label": torch.tensor([[[0, 1, 0]]])},
                    "model_context": {"offset": torch.tensor(0.0)},
                }
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
