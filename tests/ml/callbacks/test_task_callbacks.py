from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave_engine.ml.callbacks import DensePredictionLogger, ThresholdCalibrator
from geosave_engine.ml.models.contract import chain_step
from geosave_engine.ml.tasks import SemanticSegmentationTask


class SegmentationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: int | tuple[int, int],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

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


class Samples(Dataset[dict[str, dict[str, torch.Tensor]]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "layers": {
                "image": torch.ones(2, 4, 4),
                "label": torch.zeros(1, 4, 4, dtype=torch.long),
            }
        }


def test_max_steps_training_preserves_explicit_callbacks(tmp_path: Path) -> None:
    task = SemanticSegmentationTask(
        model_chain={"model": {"class_path": f"{__name__}.SegmentationModel"}},
        in_channels=2,
        num_classes=2,
        input_size=4,
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
