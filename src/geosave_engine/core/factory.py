from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseFactory(Protocol):
    """Shared interface for GeoSave factory classes."""

    doc_links: list[str] | None


@runtime_checkable
class BaseModelFactory(BaseFactory, Protocol):
    tasks: dict[str, list[str]]
    model: Any
    
    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.model(*args, **kwargs)


@runtime_checkable
class BaseLossFactory(BaseFactory, Protocol):
    loss: Any


@runtime_checkable
class BaseOptimizerFactory(BaseFactory, Protocol):
    optimizer: Any

