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
