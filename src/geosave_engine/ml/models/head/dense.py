from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from geosave_engine.ml.registry import register_model
from geosave_engine.ml.models.contract import chain_step


@register_model("head", "dense")
class DenseHead(nn.Module):
    """Map one decoded feature map to per-pixel outputs.

    Optional conv refinement, then dropout, then a 1x1 projection. Serves
    segmentation logits and pixelwise regression alike; ``num_classes`` is the
    output channel count either way.

    Args:
        num_classes: number of output channels (classes or regression outputs).
        feature_channels: channel width of the decoded feature map. Auto-wired
            from whatever fills the 'decoder' slot during stage construction; set
            it directly for standalone use.
        input_size: original input spatial size (H, W), or a single int for
            square. Auto-wired from the encoder. If the decoder's own output
            lands a few pixels off this size (patch-size quantization from the
            encoder), ``forward_logits`` resizes to match. ``None`` skips the
            correction.
        hidden_channels: width of an optional 3x3 conv-BN-ReLU refinement; ``None``
            => linear head (1x1 projection only).
        dropout: Dropout2d probability before the projection; ``0.0`` => no dropout.
    """

    def __init__(
        self,
        num_classes: int,
        feature_channels: int,
        input_size: int | tuple[int, int] | None = None,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_channels = feature_channels
        self.input_size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        layers: list[nn.Module] = []
        if hidden_channels:
            layers += [
                nn.Conv2d(
                    in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = hidden_channels
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` -> ``(B, num_classes, H, W)``."""
        return self.layers(x)

    @chain_step(head=True)
    def forward_logits(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Project feature map to per-pixel logits, resized to `input_size` if given.

        Args:
            feature_map: (B, feature_channels, H, W) decoded feature map.

        Returns:
            (B, num_classes, H, W) logits tensor.
        """
        logits = self.forward(feature_map)
        if self.input_size is not None and logits.shape[-2:] != self.input_size:
            logits = F.interpolate(
                logits,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits
