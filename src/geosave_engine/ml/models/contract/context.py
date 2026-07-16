from __future__ import annotations

from functools import wraps
from typing import Any

import torch


def model_context(requires: dict[str, type] | None = None, provides: dict[str, type] | None = None):
    """Mark a module method as a context-chain step; validate keys and types.

    The decorated method receives a ``dict[str, Any]`` context shared across
    the entire chain. Return a ``dict[str, Any]`` of new/updated keys for
    intermediate steps; ``ContextChain`` merges these into the context
    immutably before passing it to the next module.

    Terminal modules (heads) may return a ``torch.Tensor`` directly —
    ``ContextChain`` stops the chain and returns the tensor. Terminal steps
    add nothing to ctx, so leave ``provides`` unset.

    Args:
        requires: Key to expected type map. Every key must be present,
            non-None, and an instance of its declared type before the
            method body runs.
        provides: Key to expected type map this method adds to ctx on
            return. Checked against the actual returned dict on every call —
            every declared key must come back with a matching type.
            ``ContextChain`` also reads this (and ``requires``) at
            construction time to statically verify the whole chain's keys
            line up before any data ever flows through it.

    Examples:
        >>> @model_context(requires={'image': torch.Tensor}, provides={'pyramid': list})
        ... def forward_pyramid(self, ctx: dict) -> dict:
        ...     features = self.backbone(ctx['image'])
        ...     return {'pyramid': features}

        >>> @model_context(requires={'feature_map': torch.Tensor})
        ... def forward_logits(self, ctx: dict) -> torch.Tensor:
        ...     return self.head(ctx['feature_map'])
    """
    _requires: dict[str, type] = requires or {}
    _provides: dict[str, type] = provides or {}

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, ctx: dict[str, Any]) -> dict[str, Any] | torch.Tensor:
            for key, expected in _requires.items():
                if key not in ctx or ctx[key] is None:
                    raise KeyError(
                        f"{type(self).__name__}.{fn.__name__}: missing ctx['{key}']"
                    )
                if not isinstance(ctx[key], expected):
                    raise TypeError(
                        f"{type(self).__name__}.{fn.__name__}: ctx['{key}'] expected "
                        f"{expected.__name__}, got {type(ctx[key]).__name__}"
                    )

            result = fn(self, ctx)

            if isinstance(result, dict):
                for key, expected in _provides.items():
                    if key not in result:
                        raise TypeError(
                            f"{type(self).__name__}.{fn.__name__}: declared provides['{key}'] "
                            "but didn't return it"
                        )
                    if not isinstance(result[key], expected):
                        raise TypeError(
                            f"{type(self).__name__}.{fn.__name__}: provides['{key}'] expected "
                            f"{expected.__name__}, got {type(result[key]).__name__}"
                        )
            return result

        setattr(wrapper, '_is_model_context', True)
        setattr(wrapper, '_requires', _requires)
        setattr(wrapper, '_provides', _provides)
        return wrapper

    return decorator
