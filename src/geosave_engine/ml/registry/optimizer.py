from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn
from torch.optim import Adagrad, Adam, AdamW, Optimizer, RMSprop, SGD

from geosave_engine.ml.registry.factory import BuildSpec, resolve


class OptimizerSpec(BuildSpec, total=False):
    """Configure an optimizer and exact direct-child parameter groups."""

    groups: dict[str, dict[str, Any]]


OPTIMIZERS: dict[str, Callable[..., Optimizer]] = {
    "ADAMW": AdamW,
    "ADAM": Adam,
    "SGD": SGD,
    "RMSPROP": RMSprop,
    "ADAGRAD": Adagrad,
}


def build_optimizer(
    spec: OptimizerSpec,
    model: nn.Module,
    registry: Mapping[str, Callable[..., Optimizer]] = OPTIMIZERS,
) -> Optimizer:
    """Construct an optimizer over a model's trainable parameters.

    Args:
        spec: Optimizer selector, arguments, and direct-child group options.
        model: Module whose parameters will be optimized.
        registry: Named optimizer factories.

    Returns:
        Configured optimizer.

    Raises:
        TypeError: If group configuration is not a mapping.
        ValueError: If groups are unknown, overlap, or no parameters train.
    """
    unknown = set(spec) - {"name", "class_path", "init_args", "groups"}
    if unknown:
        raise ValueError(f"Unknown optimizer fields: {sorted(unknown)}")

    groups = spec.get("groups", {})
    if not isinstance(groups, Mapping):
        raise TypeError("groups must map direct child names to optimizer options")

    children = dict(model.named_children())
    unknown_groups = set(groups) - set(children)
    if unknown_groups:
        raise ValueError(f"Unknown model groups: {sorted(unknown_groups)}")

    selected: set[int] = set()
    parameter_groups: list[dict[str, Any]] = []
    for child_name, options in groups.items():
        if not isinstance(options, Mapping):
            raise TypeError(f"Optimizer group {child_name!r} options must be a mapping")
        if "params" in options:
            raise ValueError(f"Optimizer group {child_name!r} cannot override params")

        child_parameters = list(children[child_name].parameters())
        identities = {id(parameter) for parameter in child_parameters}
        if overlap := selected & identities:
            raise ValueError(
                f"Optimizer groups share parameters in {child_name!r}: "
                f"{len(overlap)} duplicate(s)"
            )
        selected.update(identities)

        trainable = [parameter for parameter in child_parameters if parameter.requires_grad]
        if trainable:
            parameter_groups.append({"params": trainable, **options})

    remaining = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in selected
    ]
    if remaining:
        parameter_groups.append({"params": remaining})
    if not parameter_groups:
        raise ValueError("Cannot build an optimizer without trainable parameters")

    selector: BuildSpec = {
        key: spec[key]
        for key in ("name", "class_path", "init_args")
        if key in spec
    }
    factory = resolve(selector, registry, Optimizer)
    return factory(parameter_groups, **spec.get("init_args", {}))
