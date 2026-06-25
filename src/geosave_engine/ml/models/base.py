from typing import Protocol, runtime_checkable


@runtime_checkable
class BackboneNormalization(Protocol):
    """Per-backbone ImageNet-style normalization stats consumed by data pipelines."""

    img_mean: list[float]
    img_std: list[float]
