from __future__ import annotations

import torch
import torch.nn as nn


class UnetSegHead(nn.Module):
    """UNet segmentation head: dropout -> 1x1 classifier.

    Shapes:
        input  : ``[B, in_channels, H, W]``
        output : ``[B, num_classes, H, W]``
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        """
        Args:
            in_channels: channel width of the decoder output.
            num_classes: number of output classes / logits.
            dropout: Dropout2d probability; ``0.0`` => no dropout.
        """
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: decoder feature ``[B, in_channels, H, W]``.

        Returns:
            Class logits ``[B, num_classes, H, W]``.
        """
        return self.classifier(self.dropout(x))


class UnetRegHead(nn.Module):
    """UNet regression head: dropout -> 1x1 projection.

    Shapes:
        input  : ``[B, in_channels, H, W]``
        output : ``[B, num_outputs, H, W]``
    """

    def __init__(self, in_channels: int, num_outputs: int = 1, dropout: float = 0.0):
        """
        Args:
            in_channels: channel width of the decoder output.
            num_outputs: number of regression channels (typically ``1``).
            dropout: Dropout2d probability; ``0.0`` => no dropout.
        """
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.projector = nn.Conv2d(in_channels, num_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: decoder feature ``[B, in_channels, H, W]``.

        Returns:
            Regression output ``[B, num_outputs, H, W]``.
        """
        return self.projector(self.dropout(x))
