from __future__ import annotations

from collections.abc import Callable, Mapping

from torch import nn

from geosave_engine.ml.registry.factory import BuildSpec, resolve

from .ohem import ProbOhemCrossEntropy2d


class CriterionSpec(BuildSpec):
    """Configure a loss module."""


CRITERIA: dict[str, Callable[..., nn.Module]] = {
    "CROSS_ENTROPY": nn.CrossEntropyLoss,
    "OHEM": ProbOhemCrossEntropy2d,
}


def build_criterion(
    spec: CriterionSpec,
    registry: Mapping[str, Callable[..., nn.Module]] = CRITERIA,
) -> nn.Module:
    """Construct a loss from a registered name or imported class.

    Args:
        spec: Loss selector and constructor arguments.
        registry: Named loss factories.

    Returns:
        Configured loss module.
    """
    factory = resolve(spec, registry, nn.Module)
    return factory(**spec.get("init_args", {}))


__all__ = [
    "CriterionSpec",
    "ProbOhemCrossEntropy2d",
    "build_criterion",
]
