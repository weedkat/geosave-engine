from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

from geosave_engine.ml.criterion import ProbOhemCrossEntropy2d
from geosave_engine.model.factory import BuildSpec


class CriterionSpec(BuildSpec):
    """Configure a criterion module."""


CRITERIA: dict[str, Callable[..., nn.Module]] = {
    "CROSS_ENTROPY": nn.CrossEntropyLoss,
    "OHEM": ProbOhemCrossEntropy2d,
}


def build_criterion(
    spec: CriterionSpec | Mapping[str, Any],
    registry: Mapping[str, Callable[..., nn.Module]] = CRITERIA,
) -> nn.Module:
    """Construct a criterion from a registered name or imported class.

    Args:
        spec: Criterion selector and constructor arguments.
        registry: Named criterion factories.

    Returns:
        Configured criterion module.
    """
    configured = CriterionSpec.model_validate(spec)
    factory = configured.resolve(registry, nn.Module)
    return factory(**configured.init_args)


__all__ = ["CriterionSpec", "build_criterion"]
