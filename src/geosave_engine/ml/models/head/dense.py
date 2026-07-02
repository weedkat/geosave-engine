from __future__ import annotations

import torch
import torch.nn as nn

import torch.nn as nn

from geosave_engine.ml.models.contract import ModelContext, model_context


class DenseHead(nn.Module):
    """General dense head: optional conv refine -> dropout -> 1x1 projection.

    Decoder-agnostic ``RasterHead`` — maps a single decoded feature map to
    per-pixel outputs (segmentation logits or pixelwise regression). Shared by
    dense tasks; ``num_classes`` is the output channel count.

    Args:
        in_channels: channel width of the decoded feature map.
        num_classes: number of output channels (classes or regression outputs).
        hidden_channels: width of an optional 3x3 conv-BN-ReLU refinement; ``None``
            => linear head (1x1 projection only).
        dropout: Dropout2d probability before the projection; ``0.0`` => no dropout.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
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

    @model_context(inputs=(['feature_map'], ['logits']))
    def forward_logits(self, ctx: ModelContext) -> ModelContext:
        """Project feature map to per-pixel outputs (logits or regression values).

        Args:
            ctx: ModelContext with ctx.inputs['feature_map'] as (B, in_channels, H, W).

        Returns:
            ModelContext with ctx.inputs = {'logits': (B, num_classes, H, W)}.
        """
        return ModelContext(
            inputs={'logits': self.forward(ctx.inputs['feature_map'])},
            sample_meta=ctx.sample_meta,
            metadata=ctx.metadata,
        )
