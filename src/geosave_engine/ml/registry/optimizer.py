from __future__ import annotations

from typing import Any

import torch.optim
from torch import nn
from torch.optim import Optimizer

from geosave_engine.ml.optimizer import STRATEGIES
from geosave_engine.ml.registry.base import BuildSpec, resolve

# Optimizer classes reachable by name, each taking per-group lr and weight_decay.
OPTIMIZERS: tuple[str, ...] = ("AdamW", "Adam", "SGD", "RMSprop", "Adagrad")

# Only where this project's default differs from the torch class's own.
OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
    "SGD": {"lr": 1e-2, "momentum": 0.9, "weight_decay": 1e-4},
    "RMSprop": {"lr": 1e-3},
}


def build_optimizer(spec: BuildSpec, model: nn.Module) -> Optimizer:
    """Construct an optimizer from a registered name or an imported class.

    A name is `"<Optimizer>"` or `"<Optimizer>.<strategy>"`, such as `"AdamW"`
    or `"SGD.freeze_encoder"`; every strategy composes with every optimizer. A
    class path instead imports the class and takes every parameter as one group.

    Args:
        spec: Name or class path with optional init_args.
        model: Module whose parameters will be optimized.

    Returns:
        Optimizer containing the selected model parameters.

    Raises:
        ValueError: The name gives no known optimizer or strategy, or the
            specification is malformed.

    Examples:
        >>> build_optimizer({"name": "AdamW", "init_args": {"lr": 1e-3}}, model)
        AdamW (...)
        >>> build_optimizer(
        ...     {"name": "SGD.split", "init_args": {"encoder_lr": 1e-4, "decoder_lr": 1e-2}},
        ...     model,
        ... )
        SGD (...)
    """
    init_args: dict[str, Any] = dict(spec.get("init_args", {}))

    if "class_path" in spec:
        imported = resolve(spec, {}, Optimizer)
        return imported(model.parameters(), **init_args)

    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty registered optimizer name")

    optimizer_name, _, strategy_name = name.partition(".")
    matched = {known.casefold(): known for known in OPTIMIZERS}
    if optimizer_name.casefold() not in matched:
        raise ValueError(
            f"Unknown optimizer {optimizer_name!r} in {name!r}; available: "
            f"{list(OPTIMIZERS)}"
        )
    if strategy_name not in STRATEGIES:
        available = [key for key in STRATEGIES if key]
        raise ValueError(
            f"Unknown strategy {strategy_name!r} in {name!r}; available: {available}"
        )

    optimizer_name = matched[optimizer_name.casefold()]
    arguments = {**OPTIMIZER_DEFAULTS.get(optimizer_name, {}), **init_args}
    return STRATEGIES[strategy_name](
        getattr(torch.optim, optimizer_name), model, **arguments
    )
