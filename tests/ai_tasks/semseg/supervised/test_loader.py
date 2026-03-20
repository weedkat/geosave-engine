from pathlib import Path

import pytest

from geosave_engine.ai_tasks.semseg.supervised.loader import DataModule


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for idx in range(1, 11):
        (images_dir / f"tile_{idx:03d}.png").touch()
        (labels_dir / f"mask_{idx:03d}.png").touch()

    (images_dir / "tile_011.png").touch()
    (images_dir / "tile_012.png").touch()

    return tmp_path


def test_data_split_returns_labeled_and_unlabeled(dataset_dir: Path):
    dm = DataModule(data_dir=dataset_dir, metadata="metadata.yaml", num_workers=0)

    train_df, val_df, test_df, unlabeled_df = dm.data_split()

    assert len(train_df) + len(val_df) + len(test_df) == 10
    assert len(unlabeled_df) == 2
    assert set(train_df.columns) == {"image", "label"}
    assert set(unlabeled_df.columns) == {"image"}


def test_prepare_data_writes_split_csvs(dataset_dir: Path):
    dm = DataModule(data_dir=dataset_dir, metadata="metadata.yaml", num_workers=0)

    dm.prepare_data()

    assert (dataset_dir / "train_split.csv").exists()
    assert (dataset_dir / "val_split.csv").exists()
    assert (dataset_dir / "test_split.csv").exists()
    assert (dataset_dir / "unlabeled_split.csv").exists()


def test_setup_fit_builds_datasets_and_dataloader(dataset_dir: Path, monkeypatch: pytest.MonkeyPatch):
    class DummyDataset:
        def __init__(self, *args):
            self.args = args

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return index

    loader_module = __import__(
        "geosave_engine.ai_tasks.semseg.supervised.loader",
        fromlist=["SemSegDataset"],
    )
    monkeypatch.setattr(loader_module, "SemSegDataset", DummyDataset)

    dm = DataModule(data_dir=dataset_dir, metadata="metadata.yaml", batch_size=8, num_workers=0)
    dm.prepare_data()
    dm.setup("fit")

    train_loader = dm.train_dataloader()

    assert hasattr(dm, "train_ds")
    assert hasattr(dm, "val_ds")
    assert train_loader.batch_size == 8


def test_invalid_ratio_raises_assertion(dataset_dir: Path):
    with pytest.raises(AssertionError):
        DataModule(
            data_dir=dataset_dir,
            metadata="metadata.yaml",
            data_split_ratio=(0.8, 0.3, 0.1),
        )
