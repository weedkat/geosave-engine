"""Inert references to values supplied when a processing stage runs."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from .base import SpecModel

_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


class Ref:
    """Name a supplied value or one of its Python attributes.

    Args:
        path: Value name followed by optional dot-separated attributes.

    Examples:
        >>> Ref("optical.gs.unpack")
        Ref('optical.gs.unpack')
    """

    __slots__ = ("_path",)

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not _PATH.fullmatch(path):
            raise ValueError(f"Invalid reference path: {path!r}")
        self._path = path

    @property
    def path(self) -> str:
        """Return the referenced Python path."""
        return self._path

    @property
    def root(self) -> str:
        """Return the name required in the supplied values."""
        return self.path.split(".", 1)[0]

    def resolve(self, values: Mapping[str, Any]) -> Any:
        """Retrieve the value, following attributes without calling methods."""
        name, *attributes = self.path.split(".")
        result = values[name]
        for attribute in attributes:
            result = getattr(result, attribute)
        return result

    def __repr__(self) -> str:
        return f"Ref({self.path!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ref) and self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)


class OperationSpec(SpecModel):
    """Declare one inert call and its literal or referenced keyword arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call: str | Ref
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("call")
    @classmethod
    def validate_call(cls, value: str | Ref) -> str | Ref:
        """Require a reference or an importable module-level call path."""
        path = value.path if isinstance(value, Ref) else value
        Ref(path)
        if isinstance(value, str) and "." not in value:
            raise ValueError("An imported call needs a module and callable name")
        return value

    @field_validator("kwargs", mode="before")
    @classmethod
    def validate_kwargs(cls, value: Any) -> Any:
        """Accept finite YAML literals and inert references without cycles."""
        return cls.validate_value(value)

    @classmethod
    def validate_value(cls, value: Any, active: set[int] | None = None) -> Any:
        """Validate nested call arguments without resolving their references."""
        if isinstance(value, Ref):
            Ref(value.path)
            return value
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        if not isinstance(value, (dict, list)):
            raise ValueError(
                f"Expected a YAML literal or Ref, got {type(value).__name__}"
            )
        active = set() if active is None else active
        if id(value) in active:
            raise ValueError("Cyclic configuration values are not supported")
        active.add(id(value))
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("Configuration mapping keys must be strings")
                return {
                    key: cls.validate_value(item, active) for key, item in value.items()
                }
            return [cls.validate_value(item, active) for item in value]
        finally:
            active.remove(id(value))

    @classmethod
    def find_references(cls, value: Any) -> tuple[Ref, ...]:
        """Return references nested in a declaration, in encounter order."""
        if isinstance(value, Ref):
            return (value,)
        if isinstance(value, dict):
            return tuple(
                reference
                for item in value.values()
                for reference in cls.find_references(item)
            )
        if isinstance(value, list):
            return tuple(
                reference for item in value for reference in cls.find_references(item)
            )
        return ()

    @property
    def references(self) -> tuple[Ref, ...]:
        """Return every call and argument reference in encounter order."""
        return self.find_references([self.call, self.kwargs])

    @classmethod
    def resolve_value(cls, value: Any, values: Mapping[str, Any]) -> Any:
        """Resolve references into fresh containers for one invocation."""
        if isinstance(value, Ref):
            return value.resolve(values)
        if isinstance(value, dict):
            return {key: cls.resolve_value(item, values) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.resolve_value(item, values) for item in value]
        return value

    def resolve_kwargs(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Build fresh keyword containers against the current named values."""
        return self.resolve_value(self.kwargs, values)
