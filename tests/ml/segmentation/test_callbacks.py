from pathlib import Path
from unittest.mock import Mock
from typing import Any

import numpy as np
import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from odc.geo.geobox import GeoBox
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave_engine.geodata import raster, stack
from geosave_engine.model.spec import ChipsSpec
from tests.ml.test_inputs import _samples
from types import SimpleNamespace
from geosave_engine.ml.segmentation.callbacks import DensePredictionLogger
from geosave_engine.ml.segmentation import supervised
from geosave_engine.ml.segmentation.calibrate import ThresholdCalibrator
from geosave_engine.model.chain import chain_step


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(2, num_classes, kernel_size=1)

    @chain_step(head=True)
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image)


class ValidationObserver(Callback):
    def __init__(self) -> None:
        self.labels: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert outputs["logits"].shape == (1, 2, 4, 4)
        self.labels.append(outputs["label"].clone())


class Samples(Dataset[tuple[dict[str, torch.Tensor], torch.Tensor, str, torch.Tensor]]):
    """One 4x4 raster read as its single tile, as `supervised.Dataset` offers it."""

    def __init__(self) -> None:
        grid = GeoBox.from_bbox((0, 0, 40, 40), "EPSG:32748", resolution=10)
        scene = raster({"red": (("y", "x"), np.ones((4, 4), "float32"))}, grid)
        samples = _samples({"a": scene}, (4, 4))
        self.reference = samples.reference
        self.spec = SimpleNamespace(chips=ChipsSpec(size=4))
        self.target = "label"
        label = raster({"label": (("y", "x"), np.zeros((4, 4), "float32"))}, grid)
        self.parents = {"a": stack({"image": scene, "label": label})}
        self.scene = scene

    def __len__(self) -> int:
        return len(self.reference)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str, torch.Tensor]:
        return (
            {"image": torch.ones(2, 4, 4)},
            torch.zeros(4, 4, dtype=torch.long),
            self.reference.id.iloc[index],
            torch.ones(4, 4, dtype=torch.bool),
        )


def test_max_steps_training_preserves_explicit_callbacks(tmp_path: Path) -> None:
    task = supervised.Module(
        model_chain={
            "model": {
                "class_path": f"{__name__}.SegmentationModel",
                "init_args": {"num_classes": 2},
            }
        },
        optimizer={"class_path": "torch.optim.SGD", "init_args": {"lr": 0.1}},
        lr_scheduler={
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
            "interval": "epoch",
        },
    )
    observer = ValidationObserver()
    image_logger = DensePredictionLogger({0: "#000000"})
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=-1,
        max_steps=1,
        val_check_interval=1,
        num_sanity_val_steps=0,
        callbacks=[observer, image_logger],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    loader = DataLoader(Samples())
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)

    assert trainer.global_step == 1
    schedule = trainer.lr_scheduler_configs[0]
    assert schedule.reduce_on_plateau
    assert schedule.monitor == "val_loss"
    assert schedule.scheduler.best < float("inf")
    callbacks = getattr(trainer, "callbacks")
    assert observer in callbacks
    assert image_logger in callbacks
    assert not any(isinstance(cb, ThresholdCalibrator) for cb in callbacks)
    assert len(observer.labels) == 1
    torch.testing.assert_close(
        observer.labels[0], torch.zeros(1, 4, 4, dtype=torch.long)
    )


@pytest.mark.parametrize("interval", [0, -1])
def test_interval_must_be_positive(interval: int) -> None:
    with pytest.raises(ValueError, match="log_image_every_n_epochs must be positive"):
        DensePredictionLogger({0: "#000000"}, log_image_every_n_epochs=interval)


@pytest.mark.parametrize("reason", ["sanity", "rank", "logger", "batch", "epoch"])
def test_skipped_batches_do_not_render_or_require_outputs(reason: str) -> None:
    callback = DensePredictionLogger({0: "#000000"})
    trainer = Mock(spec=Trainer)
    trainer.sanity_checking = reason == "sanity"
    trainer.is_global_zero = reason != "rank"
    trainer.current_epoch = 1 if reason == "epoch" else 0
    trainer.loggers = [] if reason == "logger" else [Mock(spec=TensorBoardLogger)]

    callback.on_validation_batch_end(
        trainer, LightningModule(), None, None, 1 if reason == "batch" else 0
    )
    for logger in trainer.loggers:
        logger.experiment.add_image.assert_not_called()


def test_loader_images_have_distinct_names_and_argmax_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = DensePredictionLogger({0: "#000000", 1: "#ffffff"})
    trainer = Mock(spec=Trainer)
    trainer.sanity_checking = False
    trainer.is_global_zero = True
    trainer.current_epoch = 2
    logger = Mock(spec=TensorBoardLogger)
    trainer.loggers = [logger]
    label = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.tensor([[[[3.0, 0.0], [1.0, 2.0]], [[0.0, 4.0], [2.0, 0.0]]]])
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def render(actual_label: torch.Tensor, actual_preds: torch.Tensor) -> np.ndarray:
        torch.testing.assert_close(actual_label, label[0])
        torch.testing.assert_close(actual_preds, label[0])
        return image

    monkeypatch.setattr(callback, "_render", render)
    for loader in (0, 1):
        callback.on_validation_batch_end(
            trainer,
            LightningModule(),
            {"logits": logits, "label": label},
            None,
            0,
            loader,
        )
    assert [call.args[0] for call in logger.experiment.add_image.call_args_list] == [
        "val/dataloader_0/prediction",
        "val/dataloader_1/prediction",
    ]
    assert logger.experiment.add_image.call_args.args[1] is image
    assert logger.experiment.add_image.call_args.args[2] == 2
