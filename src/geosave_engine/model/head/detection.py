from __future__ import annotations

import torch
import torch.nn as nn

from geosave_engine.model.chain import chain_step
from geosave_engine.model.registry import register_model


@register_model("head", "detection")
class DetectionHead(nn.Module):
    """Predict one box and its class scores at every cell of every pyramid level.

    A YOLO-style anchor-free head: each level has its own box branch and class
    branch. Predictions are raw; turning them into boxes on the image needs
    `pyramid_strides` and happens outside this module.

    Args:
        classes: Class names. A class's code is its position in the list.
        pyramid_channels: Channel width of each pyramid level. Wired from the
            encoder during stage construction.
        pyramid_strides: Input pixels one cell of each level spans. Wired from
            the encoder during stage construction.
        hidden_channels: Width of each branch's two 3x3 conv layers.

    Raises:
        ValueError: `classes` is empty or names a class twice, or the levels
            and strides differ in number.

    Examples:
        >>> head = DetectionHead(
        ...     classes=["tree"], pyramid_channels=[256, 512], pyramid_strides=[8, 16]
        ... )
        >>> head.num_classes
        1
    """

    def __init__(
        self,
        classes: list[str],
        pyramid_channels: list[int],
        pyramid_strides: list[int],
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        if not classes:
            raise ValueError("a detection head needs at least one class")
        repeated = sorted({name for name in classes if classes.count(name) > 1})
        if repeated:
            raise ValueError(f"classes name {repeated} more than once")
        if len(pyramid_channels) != len(pyramid_strides):
            raise ValueError(
                f"the pyramid has {len(pyramid_channels)} levels but "
                f"{len(pyramid_strides)} strides; give one stride per level"
            )
        self.classes = list(classes)
        self.pyramid_strides = list(pyramid_strides)

        def branch(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1),
            )

        self.box_branches = nn.ModuleList(
            branch(width, 4) for width in pyramid_channels
        )
        self.class_branches = nn.ModuleList(
            branch(width, len(classes)) for width in pyramid_channels
        )

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.classes)

    @chain_step(head=True)
    def forward_predictions(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        """Predict raw boxes and class scores over every level.

        Args:
            pyramid: Per-level (B, C, H, W) features from the encoder.

        Returns:
            (B, cells, 4 + num_classes) tensor, cells counted over every level
            in pyramid order. The first four values are the raw box branch
            outputs, the rest the class logits.
        """
        levels = [
            torch.cat([box(features), scores(features)], dim=1).flatten(2)
            for features, box, scores in zip(
                pyramid, self.box_branches, self.class_branches, strict=True
            )
        ]
        return torch.cat(levels, dim=2).transpose(1, 2)
