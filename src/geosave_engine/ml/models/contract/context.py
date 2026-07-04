from __future__ import annotations

from functools import wraps
from typing import Any

import torch


def model_context(requires: list[str] | None = None):
    """Mark a module method as a context-chain step and validate required keys.

    The decorated method receives a ``dict[str, Any]`` context shared across
    the entire chain. Return a ``dict[str, Any]`` of new/updated keys for
    intermediate steps; ``ContextChain`` merges these into the context
    immutably before passing it to the next module.

    Terminal modules (heads) may return a ``torch.Tensor`` directly —
    ``ContextChain`` stops the chain and returns the tensor.

    Args:
        requires: Keys that must be present and non-None in the context dict
            before the method body runs.

    Examples:
        >>> @model_context(requires=['image'])
        ... def forward_pyramid(self, ctx: dict) -> dict:
        ...     features = self.backbone(ctx['image'])
        ...     return {'pyramid': features}

        >>> @model_context(requires=['feature_map'])
        ... def forward_logits(self, ctx: dict) -> torch.Tensor:
        ...     return self.head(ctx['feature_map'])
    """
    _requires: list[str] = requires or []

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, ctx: dict[str, Any]) -> dict[str, Any] | torch.Tensor:
            for key in _requires:
                if key not in ctx or ctx[key] is None:
                    raise KeyError(
                        f"{type(self).__name__}.{fn.__name__}: missing ctx['{key}']"
                    )
            return fn(self, ctx)

        setattr(wrapper, '_is_model_context', True)
        setattr(wrapper, '_requires', _requires)
        return wrapper

    return decorator
