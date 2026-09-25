from __future__ import annotations

from collections.abc import Callable, Mapping

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from geosave_engine.ml.registry.base import BuildSpec, resolve

SCHEDULERS: dict[str, Callable[..., LRScheduler]] = {
    "CosineAnnealingLR": CosineAnnealingLR,
}


def build_scheduler(
    spec: BuildSpec,
    optimizer: Optimizer,
    registry: Mapping[str, Callable[..., LRScheduler]] = SCHEDULERS,
) -> LRScheduler:
    """Construct a scheduler for an existing optimizer.

    Args:
        spec: Name or class path with optional init_args.
        optimizer: Optimizer whose learning rate the scheduler controls.
        registry: Named scheduler factories.

    Returns:
        Scheduler attached to the supplied optimizer.
    """
    factory = resolve(spec, registry, LRScheduler)
    return factory(optimizer=optimizer, **spec.get("init_args", {}))
