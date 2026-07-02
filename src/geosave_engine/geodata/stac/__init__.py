from __future__ import annotations

from typing import TYPE_CHECKING

from .query import StacQuery

if TYPE_CHECKING:
    from .client import StacClient

__all__ = ["StacClient", "StacQuery"]


def __getattr__(name: str) -> object:
    if name == "StacClient":
        from .client import StacClient
        return StacClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
