from __future__ import annotations

import inspect
from typing import Any


def filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs not accepted by cls.__init__.

    If cls accepts **kwargs, returns all kwargs unchanged.

    Args:
        cls: Target class to inspect.
        kwargs: Candidate keyword arguments.

    Returns:
        Filtered dict containing only keys cls.__init__ accepts.
    """
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
