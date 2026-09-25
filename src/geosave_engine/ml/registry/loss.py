from __future__ import annotations

from collections.abc import Callable, Mapping

from torch import nn

from geosave_engine.ml.loss import ProbOhemCrossEntropy2d
from geosave_engine.ml.registry.base import BuildSpec, resolve

LOSSES: dict[str, Callable[..., nn.Module]] = {
    "CELoss": nn.CrossEntropyLoss,
    "OHEMLoss": ProbOhemCrossEntropy2d,
}


def build_loss(
    spec: BuildSpec,
    registry: Mapping[str, Callable[..., nn.Module]] = LOSSES,
) -> nn.Module:
    """Construct a loss from a registered factory or imported class.

    Args:
        spec: Name or class path with optional init_args.
        registry: Named loss factories.

    Returns:
        Configured loss module.
    """
    factory = resolve(spec, registry, nn.Module)
    return factory(**spec.get("init_args", {}))
