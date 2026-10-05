from __future__ import annotations

import torch
import torch.nn as nn

from geosave_engine.model.chain import chain_step
from geosave_engine.model.registry import register_model


@register_model("head", "classification")
class ClassificationHead(nn.Module):
    """Map an encoder's deepest features to one class logit vector per sample.

    Averages the deepest pyramid level over its spatial axes, then projects it
    linearly.

    Args:
        classes: Class names. A class's code is its position in the list.
        pyramid_channels: Channel width of each pyramid level. Wired from the
            encoder during stage construction.
        dropout: Dropout probability before the projection.

    Raises:
        ValueError: `classes` is empty or names a class twice.

    Examples:
        >>> head = ClassificationHead(classes=["crop", "forest"], pyramid_channels=[768] * 4)
        >>> head.num_classes
        2
    """

    def __init__(
        self,
        classes: list[str],
        pyramid_channels: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not classes:
            raise ValueError("a classification head needs at least one class")
        repeated = sorted({name for name in classes if classes.count(name) > 1})
        if repeated:
            raise ValueError(f"classes name {repeated} more than once")
        self.classes = list(classes)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(pyramid_channels[-1], len(classes))

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.classes)

    @chain_step(head=True)
    def forward_logits(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        """Project the pooled deepest level to class logits.

        Args:
            pyramid: Per-level (B, C, H, W) features from the encoder.

        Returns:
            (B, num_classes) logits tensor.
        """
        pooled = pyramid[-1].flatten(2).mean(dim=-1)
        return self.projection(self.dropout(pooled))
