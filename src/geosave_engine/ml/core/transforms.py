from __future__ import annotations

import inspect
import warnings

import torch
import torch.nn as nn
import kornia.augmentation as K

from geosave_engine.ml.models.contract import Normalization


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


class ImageProcessor(nn.Module):
    """Deterministic transforms applied on all splits: resize → normalize.

    Normalization source priority: explicit ``mean_norm``/``std_norm`` > ``model.img_mean``/``img_std`` > skip.

    Args:
        in_channels: Expected input channel count (from ``band_map``). Used to
            validate ``mean_norm``/``std_norm`` line up with the configured bands.
        model: Optional nn.Module. If it implements ``Normalization`` and no explicit
            mean/std are given, normalization stats are pulled from it.
        mean_norm: Per-channel mean override. Takes precedence over ``model``.
        std_norm: Per-channel std override. Takes precedence over ``model``.
        size: Output spatial size ``(H, W)`` or single int.

    Raises:
        ValueError: If resolved ``mean_norm``/``std_norm`` length doesn't match
            ``in_channels`` — a hardcoded model default (e.g. 3-channel ImageNet
            stats) silently mismatched against a multispectral ``band_map`` would
            otherwise fail deep inside kornia with a confusing broadcast error.
    """

    def __init__(
        self,
        in_channels: int,
        model: nn.Module | None = None,
        mean_norm: list[float] | None = None,
        std_norm: list[float] | None = None,
        size: tuple[int, int] | int | None = None,
    ) -> None:
        super().__init__()
        if mean_norm is None and std_norm is None:
            if model is not None and isinstance(model, Normalization):
                mean_norm = list(model.img_mean)
                std_norm = list(model.img_std)
            elif model is not None:
                warnings.warn(
                    f"{type(model).__name__} does not implement Normalization "
                    "(no img_mean/img_std). Normalization will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )
        if mean_norm is not None and std_norm is not None:
            if len(mean_norm) != in_channels or len(std_norm) != in_channels:
                raise ValueError(
                    f"mean_norm (len {len(mean_norm)}) / std_norm (len {len(std_norm)}) "
                    f"do not match in_channels ({in_channels}) from band_map. "
                    "Set model.init_args.mean_norm and model.init_args.std_norm to "
                    f"lists of length {in_channels}, one value per band in band_map."
                )
        self.resize = K.Resize(size) if size is not None else nn.Identity()
        self.normalize: K.Normalize | None = (
            K.Normalize(mean=torch.tensor(mean_norm), std=torch.tensor(std_norm))
            if mean_norm is not None and std_norm is not None
            else None
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """resize → normalize (if configured). Input ``[B, C, H, W]``."""
        img = self.resize(img)
        if self.normalize is not None:
            img = self.normalize(img)
        return img


class ImageAugmenter(nn.Module):
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