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

from geosave_engine.ml.registry.factory import BuildSpec, resolve


class SchedulerSpec(BuildSpec, total=False):
    """Configure a scheduler and its Lightning metadata."""

    interval: Literal["step", "epoch"]
    frequency: int
    monitor: str
    strict: bool
    scheduler_name: str


SCHEDULERS: dict[str, Callable[..., LRScheduler]] = {
    "COSINE_ANNEALING": CosineAnnealingLR,
    "REDUCE_ON_PLATEAU": ReduceLROnPlateau,
    "STEP": StepLR,
}


def build_scheduler(
    spec: SchedulerSpec,
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
    metadata_fields = {"interval", "frequency", "monitor", "strict"}
    allowed = {"name", "class_path", "init_args", "scheduler_name", *metadata_fields}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown scheduler fields: {sorted(unknown)}")

    selector: BuildSpec = {
        key: spec[key]
        for key in ("name", "class_path", "init_args")
        if key in spec
    }
    factory = resolve(selector, registry, LRScheduler)
    scheduler = factory(optimizer, **spec.get("init_args", {}))
    configured = {
        key: spec[key]
        for key in metadata_fields
        if key in spec
    }
    if "scheduler_name" in spec:
        configured["name"] = spec["scheduler_name"]
    return {"scheduler": scheduler, **configured}
