from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from geosave_engine.ml.data import SemanticSegmentationDataModule


class Samples(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        image = torch.full((2, 3, 3), index, dtype=torch.float32)
        target = torch.full((3, 3), index, dtype=torch.int64)
        return {"image": image}, target


class DataModule(SemanticSegmentationDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.requests: list[tuple[Path, bool]] = []

    def _dataset(self, path: Path, *, training: bool) -> Dataset:
        self.requests.append((path, training))
        return Samples()


def test_data_module_leaves_the_supervised_dataset_explicit() -> None:
    data = SemanticSegmentationDataModule(
        train_path="train",
        val_path="val",
        input_size=32,
        batch_size=4,
        num_workers=0,
    )

    assert isinstance(data, LightningDataModule)
    assert data.input_size == (32, 32)
    with pytest.raises(NotImplementedError, match="supervised Dataset"):
        data.setup("fit")


def test_data_module_builds_torch_datasets_during_lightning_setup() -> None:
    data = DataModule(
        train_path="train",
        val_path="val",
        input_size=(3, 3),
        batch_size=2,
        num_workers=0,
    )

    data.setup("fit")

    assert data.requests == [(Path("train"), True), (Path("val"), False)]
    train = data.train_dataloader()
    validation = data.val_dataloader()
    assert isinstance(train, DataLoader)
    assert isinstance(validation, DataLoader)
    inputs, target = next(iter(validation))
    assert inputs["image"].shape == (2, 2, 3, 3)
    assert target.shape == (2, 3, 3)
