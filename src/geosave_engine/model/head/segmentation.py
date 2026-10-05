from __future__ import annotations

from geosave_engine.model.registry import register_model

from .dense import DenseHead


@register_model("head", "segmentation")
class SegmentationHead(DenseHead):
    """Map one decoded feature map to per-pixel class logits.

    Args:
        classes: Class names. A class's code is its position in the list.
        feature_channels: Channel width of the decoded feature map. Wired from
            the decoder during stage construction.
        input_size: Input spatial size the logits are resized to. Wired from
            the encoder. None leaves the decoder's size.
        hidden_channels: Width of an optional 3x3 conv refinement. None
            projects linearly.
        dropout: Dropout2d probability before the projection.

    Raises:
        ValueError: `classes` is empty or names a class twice.

    Examples:
        >>> head = SegmentationHead(classes=["background", "oil_palm"], feature_channels=256)
        >>> head.num_classes
        2
    """

    def __init__(
        self,
        classes: list[str],
        feature_channels: int,
        input_size: int | tuple[int, int] | None = None,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        if not classes:
            raise ValueError("a segmentation head needs at least one class")
        repeated = sorted({name for name in classes if classes.count(name) > 1})
        if repeated:
            raise ValueError(f"classes name {repeated} more than once")
        super().__init__(
            num_classes=len(classes),
            feature_channels=feature_channels,
            input_size=input_size,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.classes = list(classes)

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.classes)
