"""Inert references to values supplied when a processing stage runs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import math
import re
from typing import Any

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


def validate_value(value: Any, active: set[int] | None = None) -> Any:
    """Accept only finite YAML literals and inert references, without cycles."""
    if isinstance(value, Ref):
        Ref(value.path)
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if not isinstance(value, (dict, list)):
        raise ValueError(f"Expected a YAML literal or Ref, got {type(value).__name__}")
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("Cyclic configuration values are not supported")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Configuration mapping keys must be strings")
            return {key: validate_value(item, active) for key, item in value.items()}
        return [validate_value(item, active) for item in value]
    finally:
        active.remove(id(value))


def references(value: Any) -> Iterator[Ref]:
    """Walk configuration references, never the contents of resolved objects."""
    if isinstance(value, Ref):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def resolve(value: Any, values: Mapping[str, Any]) -> Any:
    """Build fresh argument containers while passing referenced objects unchanged."""
    if isinstance(value, Ref):
        return value.resolve(values)
    if isinstance(value, dict):
        return {key: resolve(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, values) for item in value]
    return value
