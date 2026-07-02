from __future__ import annotations

import torch.nn as nn

from geosave_engine.ml.models.contract.context import ModelContextSpec


def _get_context_specs(cls: type) -> dict[str, ModelContextSpec]:
    """Scan a class's MRO for all @model_context decorated methods.

    Args:
        cls: The class to inspect.

    Returns:
        {
            method_name: ModelContextSpec,
            ...
        }
    """
    specs: dict[str, ModelContextSpec] = {}
    for klass in reversed(cls.__mro__):
        for name, val in vars(klass).items():
            if callable(val) and hasattr(val, '_model_context_spec'):
                specs[name] = val._model_context_spec
    return specs


def _discover_chain(modules: list[nn.Module]) -> list[tuple[nn.Module, str]]:
    """Resolve the single @model_context method on each module.

    Each module must declare exactly one @model_context method.
    If a module has none or multiple, the chain is ambiguous and construction fails.

    Args:
        modules: Ordered list of nn.Module instances forming the pipeline.

    Returns:
        List of (module, method_name) pairs in the same order.

    Raises:
        TypeError: If any module has 0 or more than 1 @model_context methods.
    """
    chain: list[tuple[nn.Module, str]] = []
    for module in modules:
        specs = _get_context_specs(type(module))
        if not specs:
            raise TypeError(
                f"{type(module).__name__}: no @model_context method found. "
                "Every module in a chain must declare exactly one."
            )
        if len(specs) > 1:
            raise TypeError(
                f"{type(module).__name__}: ambiguous — multiple @model_context methods "
                f"{sorted(specs)}. Chain requires exactly one per module."
            )
        chain.append((module, next(iter(specs))))
    return chain


