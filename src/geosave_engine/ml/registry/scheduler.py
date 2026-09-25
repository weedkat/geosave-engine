from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

from geosave_engine.ml.registry.factory import BuildSpec


class SchedulerSpec(BuildSpec):
    """Configure a scheduler and its Lightning metadata."""

    interval: Literal["step", "epoch"] | None = None
    frequency: int | None = None
    monitor: str | None = None
    strict: bool | None = None
    scheduler_name: str | None = None


SCHEDULERS: dict[str, Callable[..., LRScheduler]] = {
    "COSINE_ANNEALING": CosineAnnealingLR,
    "REDUCE_ON_PLATEAU": ReduceLROnPlateau,
    "STEP": StepLR,
}


def build_scheduler(
    spec: SchedulerSpec | Mapping[str, Any],
    optimizer: Optimizer,
    registry: Mapping[str, Callable[..., LRScheduler]] = SCHEDULERS,
) -> dict[str, Any]:
    """Construct a scheduler and its Lightning configuration mapping.

    Args:
        spec: Scheduler selector, arguments, and Lightning metadata.
        optimizer: Optimizer controlled by the scheduler.
        registry: Named scheduler factories.

    Returns:
        Lightning scheduler configuration.
    """
    configured = SchedulerSpec.model_validate(spec)
    factory = configured.resolve(registry, LRScheduler)
    scheduler = factory(optimizer, **configured.init_args)
    metadata = configured.model_dump(
        include={"interval", "frequency", "monitor", "strict"},
        exclude_none=True,
    )
    if configured.scheduler_name is not None:
        metadata["name"] = configured.scheduler_name
    return {"scheduler": scheduler, **metadata}
