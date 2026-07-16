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


def _validate_chain(chain: list[tuple[nn.Module, str]]) -> None:
    """Statically verify every non-first step's ``requires`` is satisfied before any data flows.

    Walks the chain in order, accumulating each step's declared ``provides``.
    From the second step on, ``requires`` must be a subset of everything
    accumulated so far, with matching declared types — catches a wrong
    module order or a typo'd key at construction time instead of a
    ``KeyError``/``TypeError`` on the first real forward call.

    The first step is not checked here — its ``requires`` comes from outside
    the chain entirely (whatever the caller passes to ``forward()``), which
    isn't known at construction time. See ``ContextChain.required_keys`` to
    read what the first step expects.

    Args:
        chain: Ordered (module, method_name) pairs from ``_discover_chain``.

    Raises:
        TypeError: A non-first step requires a key nothing earlier provides,
            or the declared type doesn't match what was provided.
    """
    available: dict[str, type] = {}
    for i, (module, method_name) in enumerate(chain):
        method = getattr(module, method_name)
        requires: dict[str, type] = getattr(method, '_requires', {})
        provides: dict[str, type] = getattr(method, '_provides', {})
        if i > 0:
            for key, expected in requires.items():
                if key not in available:
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}: requires ctx['{key}'] but no "
                        "earlier step provides it"
                    )
                actual = available[key]
                if not issubclass(actual, expected):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}: requires ctx['{key}'] as "
                        f"{expected.__name__}, but {actual.__name__} was provided earlier"
                    )
        available.update(provides)


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
        TypeError: If any module has 0 or more than 1 @model_context methods,
            or the chain's requires/provides don't line up from the second
            step on (see ``_validate_chain``).

    Examples:
        >>> chain = ContextChain({'encoder': enc, 'decoder': dec, 'head': hd})
        >>> chain.required_keys  # {'image': torch.Tensor} -- what forward() needs
        >>> logits = chain({'image': x})  # enc → dec → hd; head returns Tensor
    """

    def __init__(self, modules: dict[str, nn.Module]) -> None:
        super().__init__()
        for name, mod in modules.items():
            self.add_module(name, mod)
        self._chain = _discover_chain(list(modules.values()))
        _validate_chain(self._chain)

    @property
    def required_keys(self) -> dict[str, type]:
        """Keys the first step needs in ``forward()``'s ctx.

        Not satisfiable from within the chain — the caller (whatever builds
        ctx before calling this chain) must supply these directly. Empty for
        an empty chain.
        """
        if not self._chain:
            return {}
        module, method_name = self._chain[0]
        return dict(getattr(getattr(module, method_name), '_requires', {}))

    def forward(self, ctx: dict[str, Any]) -> dict[str, Any] | torch.Tensor:
        for module, method_name in self._chain:
            result = getattr(module, method_name)(ctx)
            if isinstance(result, torch.Tensor):
                return result
            ctx = {**ctx, **result}
        return ctx
