from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import kornia.augmentation as K

from geosave_engine.ml.models.contract import Normalization


class ImageProcessor(nn.Module):
    """Deterministic transforms applied on all splits: resize → normalize.

    Normalization source priority: explicit ``mean_norm``/``std_norm`` > ``model.img_mean``/``img_std`` > skip.

    Args:
        model: Optional nn.Module. If it implements ``Normalization`` and no explicit
            mean/std are given, normalization stats are pulled from it.
        mean_norm: Per-channel mean override. Takes precedence over ``model``.
        std_norm: Per-channel std override. Takes precedence over ``model``.
        size: Output spatial size ``(H, W)`` or single int.

    Raises:
        ValueError: If resolved ``mean_norm``/``std_norm`` have mismatched lengths
            (init), or a `forward()` input's channel count doesn't match resolved
            ``mean_norm``/``std_norm`` (a hardcoded model default, e.g. 3-channel
            ImageNet stats, silently mismatched against a multispectral
            ``band_map`` would otherwise fail deep inside kornia with a confusing
            broadcast error).
    """

    def __init__(
        self,
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
        if mean_norm is not None and std_norm is not None and len(mean_norm) != len(std_norm):
            raise ValueError(f"mean_norm (len {len(mean_norm)}) and std_norm (len {len(std_norm)}) must match.")
        self.resize = K.Resize(size) if size is not None else nn.Identity()
        self.normalize: K.Normalize | None = (
            K.Normalize(mean=torch.tensor(mean_norm), std=torch.tensor(std_norm))
            if mean_norm is not None and std_norm is not None
            else None
        )
        self._expected_channels = len(mean_norm) if mean_norm is not None else None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """resize → normalize (if configured). Input ``[B, C, H, W]``.

        Casts to float first — kornia's resize/normalize both require
        float16/32/64, regardless of the input's real (e.g. int) dtype.
        """
        img = img.float()
        img = self.resize(img)
        if self.normalize is not None:
            if img.shape[1] != self._expected_channels:
                raise ValueError(
                    f"input has {img.shape[1]} channel(s), but configured mean_norm/std_norm "
                    f"expect {self._expected_channels} — check band_map against model/mean_norm/std_norm."
                )
            img = self.normalize(img)
        return img
