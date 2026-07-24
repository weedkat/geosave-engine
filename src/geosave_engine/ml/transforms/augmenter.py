from __future__ import annotations

import inspect

import torch
import kornia.augmentation as K


def build_augmentation_pipeline(augmentations: list[dict], size: tuple[int, int] | int) -> list:
    """Build list of Kornia augmentation instances from config dicts.

    Recursively handles ``AugmentationSequential``. Injects ``size`` into any
    augmentation whose ``__init__`` accepts it, if not already set.

    Args:
        augmentations: List of dicts with ``"name"`` and optional ``"init_args"``.
        size: Default spatial size injected into size-aware augmentations.

    Returns:
        List of instantiated Kornia augmentation objects.

    Raises:
        ValueError: If ``name`` is not a valid Kornia augmentation class.
    """
    aug_instances = []
    for aug in augmentations:
        aug_name = aug["name"]
        init_args = aug.get("init_args", {}).copy()
        if not hasattr(K, aug_name):
            raise ValueError(f"Unknown augmentation '{aug_name}' in Kornia.")
        aug_cls = getattr(K, aug_name)
        if aug_name == "AugmentationSequential":
            nested = build_augmentation_pipeline(aug.get("augmentations", []), size)
            aug_instances.append(K.AugmentationSequential(*nested, **init_args))
            continue
        sig = inspect.signature(aug_cls.__init__)
        if "size" in sig.parameters:
            init_args.setdefault("size", size)
        aug_instances.append(aug_cls(**init_args))
    return aug_instances


class ImageAugmenter(torch.nn.Module):
    """Stochastic Kornia transforms for training. Applies same transform to image, label, and nodata mask.

    Args:
        augmentations: List of augmentation config dicts (``"name"``, ``"init_args"``).
        size: Default size injected into size-aware augmentations.
    """

    def __init__(
        self,
        augmentations: list[dict],
        size: tuple[int, int] | int,
    ) -> None:
        super().__init__()
        aug_list = build_augmentation_pipeline(augmentations, size)
        # data_keys order: image, label, nodata_mask
        self.pipeline: K.AugmentationSequential | None = (
            K.AugmentationSequential(*aug_list, data_keys=["input", "mask"])
            if aug_list else None
        )

    def forward(
        self,
        img: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply stochastic pipeline to image, label, and optional nodata mask.

        All three tensors receive the same spatial transform so label
        stay aligned with the augmented image.

        Args:
            img: Float image ``[B, C, H, W]``.
            label: Integer label ``[B, 1, H, W]``.

        Returns:
            Tuple of ``(augmented_img, augmented_label)``, label still ``[B, 1, H, W]``.
        """
        if self.pipeline is not None:
            img, label = self.pipeline(img, label)
        return img, label
