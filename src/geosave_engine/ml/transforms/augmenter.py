"""Construct native Kornia augmentation from training configuration."""

from __future__ import annotations

import inspect
from typing import Any

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase


def build_augmentation_pipeline(
    augmentations: list[dict[str, Any]], size: tuple[int, int] | int
) -> list[_AugmentationBase | K.ImageSequential]:
    """Construct Kornia transforms, using the tile size where none is given.

    Nested `AugmentationSequential` configs retain their own constructor options.
    The configuration is left unchanged.
    """
    shape = (size, size) if isinstance(size, int) else size
    transforms = []
    for step in augmentations:
        try:
            augmentation = getattr(K, step["name"])
        except AttributeError as error:
            raise ValueError(f"Unknown Kornia augmentation {step['name']!r}") from error
        options = dict(step.get("init_args", {}))
        if step["name"] == "AugmentationSequential":
            transforms.append(
                K.AugmentationSequential(
                    *build_augmentation_pipeline(step.get("augmentations", []), shape),
                    **options,
                )
            )
        else:
            if "size" in inspect.signature(augmentation).parameters:
                options.setdefault("size", shape)
            transforms.append(augmentation(**options))
    return transforms


class ImageAugmenter(K.AugmentationSequential):
    """Configure native Kornia augmentation from YAML `name`/`init_args` entries.

    Args:
        augmentations: Kornia transforms and their constructor arguments.
        size: Default spatial size for transforms such as RandomCrop.
        data_keys: Native Kornia keys matching the supplied tensors. Defaults
            to one image; supervised segmentation supplies images and a mask.

    Raises:
        ValueError: A transform name or native Kornia data key is unknown.
    """

    def __init__(
        self,
        augmentations: list[dict[str, Any]],
        size: tuple[int, int] | int,
        data_keys: list[str] | None = None,
    ) -> None:
        super().__init__(
            *build_augmentation_pipeline(augmentations, size),
            data_keys=data_keys or ["input"],
        )
