from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
)

from geosave_engine.ml.registry.base import builder

SCHEDULERS = {
    "LRScheduler": LRScheduler,
    "CosineAnnealingLR": CosineAnnealingLR,
}


def build_scheduler(name: str, optimizer: Optimizer, config: dict, registry: dict = SCHEDULERS) -> LRScheduler | None:
    """Build LR scheduler by name from registry.

    Args:
        name: Registry key (e.g. ``"CosineAnnealingLR"``).
        optimizer: Optimizer passed to scheduler constructor.
        config: Keyword args passed to the scheduler constructor.
        registry: Mapping of name → scheduler class. Defaults to ``SCHEDULERS``.

    Returns:
        Instantiated ``LRScheduler``, or ``None`` if ``name`` is ``None``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return builder(name, {**config, "optimizer": optimizer}, registry)
