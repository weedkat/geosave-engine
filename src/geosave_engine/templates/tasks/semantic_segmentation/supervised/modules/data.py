"""Load prepared image/target pairs for this segmentation workspace."""

from functools import partial

from lightning import LightningDataModule
from litdata import StreamingDataLoader, StreamingDataset
import torch
from torch.utils.data import default_collate

from geosave_engine.ml.transforms import ImageAugmenter


class SegmentationDataModule(LightningDataModule):
    """Stream prepared ``(image, target)`` tensors and augment training batches.

    Args:
        train_path: LitData directory containing training items.
        val_path: LitData directory containing validation items.
        input_size: Spatial size supplied to size-aware augmentations.
        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers.
        augmentations: Paired Kornia transforms applied only during training.
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        input_size: int | tuple[int, int] = 224,
        batch_size: int = 8,
        num_workers: int = 4,
        augmentations: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.input_size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmenter = ImageAugmenter(
            augmentations or [], size=self.input_size, data_keys=["image", "mask"]
        )

    def setup(self, stage: str | None = None) -> None:
        """Open the prepared streams needed by fitting or validation."""
        if stage in (None, "fit"):
            self.train = StreamingDataset(self.train_path, shuffle=True)
        if stage in (None, "fit", "validate"):
            self.val = StreamingDataset(self.val_path, shuffle=False)

    def collate(
        self, samples: list, training: bool = True
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Batch decoded pairs, keeping image and mask transforms aligned."""
        image, target = default_collate(samples)
        if training:
            # Kornia masks have a channel dimension; retain the prepared layout.
            squeezed = target.ndim == 3
            mask = target.unsqueeze(1) if squeezed else target
            image, mask = self.augmenter(image, mask)
            target = mask.squeeze(1) if squeezed else mask
        return {"image": image}, target

    def train_dataloader(self) -> StreamingDataLoader:
        """Return shuffled training batches with paired augmentation."""
        return StreamingDataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def val_dataloader(self) -> StreamingDataLoader:
        """Return validation batches without stochastic augmentation."""
        return StreamingDataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(self.collate, training=False),
        )
