from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseInterface(Protocol):
    """Shared interface for GeoSave registry/discovery classes."""

    doc_links: list[str] | str | None


@runtime_checkable
class ModelInterface(BaseInterface, Protocol):
    tasks: dict[str, list[str]]
    model: Any
    
    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class LossInterface(BaseInterface, Protocol):
    loss: Any

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class OptimizerInterface(BaseInterface, Protocol):
    optimizer: Any

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> Any:
        ...

