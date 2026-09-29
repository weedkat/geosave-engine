from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SemanticSegmentationDataModule(LightningDataModule):
    """Load supervised segmentation samples through PyTorch datasets.

    Subclasses implement :meth:`_dataset` for their storage format and return
    samples shaped as ``(model_inputs, target)``.

    Args:
        train_path: Training dataset location.
        val_path: Validation dataset location.
        input_size: Expected spatial input size.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
    """

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        input_size: int | tuple[int, int] = 224,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.input_size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _dataset(self, path: Path, *, training: bool) -> Dataset:
        """Build the dataset for one split."""
        raise NotImplementedError(
            "SemanticSegmentationDataModule supervised Dataset is not implemented yet"
        )

    def setup(self, stage: str | None = None) -> None:
        """Build datasets required by the requested Lightning stage."""
        if stage in (None, "fit"):
            self.train_dataset = self._dataset(self.train_path, training=True)
        if stage in (None, "fit", "validate"):
            self.val_dataset = self._dataset(self.val_path, training=False)

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        """Return shuffled training batches."""
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation batches in dataset order."""
        return self._loader(self.val_dataset, shuffle=False)
