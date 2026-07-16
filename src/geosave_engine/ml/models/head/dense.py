from __future__ import annotations

import torch
import torch.nn as nn

from geosave_engine.ml.core.factory import register_model
from geosave_engine.ml.models.contract import model_context


@register_model('head', 'dense')
class DenseHead(nn.Module):
    """General dense head: optional conv refine -> dropout -> 1x1 projection.

    Decoder-agnostic ``RasterHead`` — maps a single decoded feature map to
    per-pixel outputs (segmentation logits or pixelwise regression). Shared by
    dense tasks; ``num_classes`` is the output channel count.

    Args:
        in_channels: channel width of the decoded feature map. Set this
            directly for standalone use; leave it ``None`` and set
            ``decoder_out_channels`` instead when built through
            ``build_model`` (auto-wired from whatever fills the 'decoder'
            slot) — ``in_channels`` wins if both are given.
        decoder_out_channels: same value as ``in_channels``, named for
            ``build_model``'s stage-based auto-wiring (see ``_stage_kwargs``).
        num_classes: number of output channels (classes or regression outputs).
        hidden_channels: width of an optional 3x3 conv-BN-ReLU refinement; ``None``
            => linear head (1x1 projection only).
        dropout: Dropout2d probability before the projection; ``0.0`` => no dropout.

    Raises:
        ValueError: Neither ``in_channels`` nor ``decoder_out_channels`` given.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int | None = None,
        decoder_out_channels: int | None = None,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        resolved_in_channels = in_channels if in_channels is not None else decoder_out_channels
        if resolved_in_channels is None:
            raise ValueError("DenseHead requires 'in_channels' or 'decoder_out_channels'")
        in_channels = resolved_in_channels
        layers: list[nn.Module] = []
        if hidden_channels:
            layers += [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
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

    @model_context(requires={'feature_map': torch.Tensor})
    def forward_logits(self, ctx: dict) -> torch.Tensor:
        """Project feature map to per-pixel logits.

        Args:
            ctx: Context dict with 'feature_map' as (B, in_channels, H, W).

        Returns:
            (B, num_classes, H, W) logits tensor.
        """
        return self.forward(ctx['feature_map'])
