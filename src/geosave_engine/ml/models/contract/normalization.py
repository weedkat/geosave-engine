from typing import Protocol, runtime_checkable

@runtime_checkable
class Normalization(Protocol):

    img_mean: list[float]
    img_std: list[float]