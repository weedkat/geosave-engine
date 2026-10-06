from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import Field
from torch import nn
from torch.optim import Adagrad, Adam, AdamW, Optimizer, RMSprop, SGD

from geosave_engine.model.factory import BuildSpec


class OptimizerSpec(BuildSpec):
    """Configure an optimizer and exact direct-child parameter groups."""

    groups: dict[str, dict[str, Any]] = Field(default_factory=dict)


OPTIMIZERS: dict[str, Callable[..., Optimizer]] = {
    "ADAMW": AdamW,
    "ADAM": Adam,
    "SGD": SGD,
    "RMSPROP": RMSprop,
    "ADAGRAD": Adagrad,
}


def build_optimizer(
    spec: OptimizerSpec | Mapping[str, Any],
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
        KeyError: If a group is not a direct child of the model.
        ValueError: If groups overlap, override params, or no parameters train.
    """
    configured = OptimizerSpec.model_validate(spec)
    groups = configured.groups

    children = dict(model.named_children())
    unknown_groups = set(groups) - set(children)
    if unknown_groups:
        raise KeyError(f"Unknown model groups: {sorted(unknown_groups)}")

    selected: set[int] = set()
    parameter_groups: list[dict[str, Any]] = []
    for child_name, options in groups.items():
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

    factory = configured.resolve(registry, Optimizer)
    return factory(parameter_groups, **configured.init_args)
