from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _discover_chain(modules: list[nn.Module]) -> list[tuple[nn.Module, str]]:
    """Resolve the single @model_context method on each module.

    Each module must declare exactly one @model_context method.
    0 or multiple raises immediately — an ambiguous chain is a config error.

    Args:
        modules: Ordered list of nn.Module instances forming the pipeline.

    Returns:
        List of (module, method_name) pairs in the same order.

    Raises:
        TypeError: If any module has 0 or more than 1 @model_context methods.
    """
    chain: list[tuple[nn.Module, str]] = []
    for module in modules:
        seen: set[str] = set()
        methods: list[str] = []
        for klass in reversed(type(module).__mro__):
            for name, val in vars(klass).items():
                if callable(val) and getattr(val, '_is_model_context', False) and name not in seen:
                    methods.append(name)
                    seen.add(name)
        if not methods:
            raise TypeError(
                f"{type(module).__name__}: no @model_context method found. "
                "Every module in a chain must declare exactly one."
            )
        if len(methods) > 1:
            raise TypeError(
                f"{type(module).__name__}: ambiguous — multiple @model_context methods "
                f"{sorted(methods)}. Chain requires exactly one per module."
            )
        chain.append((module, methods[0]))
    return chain


class ContextChain(nn.Module):
    """nn.Module that registers submodules and chains their @model_context methods.

    Takes an ordered dict of name → module, registers each as a named submodule,
    discovers the single @model_context method per module, then runs the chain in
    order on each forward call.

    Each intermediate module receives the shared ``dict[str, Any]`` context and
    returns a dict of its outputs. These are merged immutably into the context
    (``ctx = {**ctx, **result}``) before the next module runs — prior keys are
    preserved without mutation, so branching and intermediate inspection are safe.

    A terminal module (typically a head) may return a ``torch.Tensor`` directly
    to end the chain early.

    Args:
        modules: Ordered dict mapping attribute name to module.
            Example: ``{'encoder': enc, 'decoder': dec, 'head': hd}``
            Example: ``{'model': monolith}``

    Raises:
        TypeError: If any module has 0 or more than 1 @model_context methods.

    Examples:
        >>> chain = ContextChain({'encoder': enc, 'decoder': dec, 'head': hd})
        >>> logits = chain({'image': x})  # enc → dec → hd; head returns Tensor
    """

    def __init__(self, modules: dict[str, nn.Module]) -> None:
        super().__init__()
        for name, mod in modules.items():
            self.add_module(name, mod)
        self._chain = _discover_chain(list(modules.values()))

    def forward(self, ctx: dict[str, Any]) -> dict[str, Any] | torch.Tensor:
        for module, method_name in self._chain:
            result = getattr(module, method_name)(ctx)
            if isinstance(result, torch.Tensor):
                return result
            ctx = {**ctx, **result}
        return ctx
