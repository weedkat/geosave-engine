"""Inert references and complete model-owned call declarations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
import math
import re

from pydantic import ConfigDict, Field, field_validator

from .base import SpecModel

_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


class Ref:
    """Name a supplied value or one of its Python attributes.

    Args:
        path: Value name followed by optional dot-separated attributes.
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
        """Return the name required in the supplied inputs."""
        return self.path.split(".", 1)[0]

    def resolve(self, inputs: Mapping[str, object]) -> object:
        """Retrieve the value, following attributes without calling methods."""
        name, *attributes = self.path.split(".")
        result = inputs[name]
        for attribute in attributes:
            result = getattr(result, attribute)
        return result

    def __repr__(self) -> str:
        return f"Ref({self.path!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ref) and self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)


type CallValue = (
    None
    | bool
    | int
    | float
    | str
    | Ref
    | list[CallValue]
    | dict[str, CallValue]
)


class CallSpec(SpecModel):
    """Declare one inert call and its literal or referenced keyword arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call: str | Ref
    kwargs: dict[str, CallValue] = Field(default_factory=dict)

    @field_validator("call")
    @classmethod
    def validate_call(cls, value: str | Ref) -> str | Ref:
        """Require a reference or an importable module-level call path."""
        if isinstance(value, Ref):
            return value
        if not _PATH.fullmatch(value):
            raise ValueError(f"Invalid call path: {value!r}")
        if "." not in value:
            raise ValueError("An imported call needs a module and callable name")
        return value

    @field_validator("kwargs", mode="before")
    @classmethod
    def validate_kwargs(cls, value: object) -> object:
        """Accept finite YAML literals and inert references without cycles."""
        return cls._validate_value(value)

    @classmethod
    def _validate_value(
        cls, value: object, active: set[int] | None = None
    ) -> object:
        if value is None or isinstance(value, (Ref, str, bool, int)):
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
                    key: cls._validate_value(item, active)
                    for key, item in value.items()
                }
            return [cls._validate_value(item, active) for item in value]
        finally:
            active.remove(id(value))

    @classmethod
    def _find_references(cls, value: object) -> tuple[Ref, ...]:
        if isinstance(value, Ref):
            return (value,)
        if isinstance(value, dict):
            return tuple(
                reference
                for item in value.values()
                for reference in cls._find_references(item)
            )
        if isinstance(value, list):
            return tuple(
                reference
                for item in value
                for reference in cls._find_references(item)
            )
        return ()

    @property
    def references(self) -> tuple[Ref, ...]:
        """Return every call and argument reference in encounter order."""
        return self._find_references([self.call, self.kwargs])

    @property
    def inputs(self) -> frozenset[str]:
        """Return the runtime roots required by this declaration."""
        return frozenset(reference.root for reference in self.references)

    def select_inputs(self, state: Mapping[str, object]) -> dict[str, object]:
        """Select only referenced roots from the current stage state."""
        if missing := self.inputs - state.keys():
            raise ValueError(f"Missing call inputs: {sorted(missing)}")
        return {
            reference.root: state[reference.root] for reference in self.references
        }

    @classmethod
    def _resolve_value(
        cls, value: CallValue, inputs: Mapping[str, object]
    ) -> object:
        if isinstance(value, Ref):
            return value.resolve(inputs)
        if isinstance(value, dict):
            return {
                key: cls._resolve_value(item, inputs) for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._resolve_value(item, inputs) for item in value]
        return value

    def _resolve_target(
        self, inputs: Mapping[str, object]
    ) -> Callable[..., object]:
        if isinstance(self.call, Ref):
            target = self.call.resolve(inputs)
        else:
            module, name = self.call.rsplit(".", 1)
            target = getattr(import_module(module), name)
        if not callable(target):
            raise TypeError(f"Expected a callable, got {type(target).__name__}")
        return target

    def invoke(self, inputs: Mapping[str, object], /) -> object:
        """Resolve and invoke this declaration exactly once."""
        try:
            target = self._resolve_target(inputs)
            kwargs = {
                name: self._resolve_value(value, inputs)
                for name, value in self.kwargs.items()
            }
            return target(**kwargs)
        except Exception as error:
            error.add_note(f"While invoking {self.call!r}")
            raise
