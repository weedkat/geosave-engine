"""Importable Dataset operations and named raster recipes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from importlib import import_module
import inspect
from operator import attrgetter
from typing import Annotated, Any, Self

from pydantic import Field, JsonValue, model_validator

from .base import Name, RasterName, SpecModel, Text, unique


class OperationSpec(SpecModel):
    """Invoke a native method or installed Python on the current raster.

    Args:
        method: Dataset/accessor method, such as gs.unpack or assign_coords.
        call: Custom synchronous callable import path, instead of method.
        kwargs: Primitive constant keyword arguments.
        inputs: Keyword argument names mapped to other rasters in the stack.
    """

    method: Text | None = None
    call: Text | None = None
    kwargs: dict[Name, JsonValue] = Field(default_factory=dict)
    inputs: dict[Name, RasterName] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_call(self) -> Self:
        if (self.call is None) == (self.method is None):
            raise ValueError("Specify exactly one of method or call")
        if self.method is not None and not all(
            part.isidentifier() for part in self.method.split(".")
        ):
            raise ValueError("Operation method must be a plain attribute path")
        if collision := self.kwargs.keys() & self.inputs.keys():
            raise ValueError(f"Operation arguments collide: {sorted(collision)}")
        if self.call is not None:
            self.resolve()
        return self

    def _import_call(self) -> Callable:
        """Resolve a custom function from its installed import path."""
        assert self.call is not None
        parts = self.call.split(".")
        if len(parts) < 2 or not all(part.isidentifier() for part in parts):
            raise ValueError("Operation call must be a dotted Python import path")
        target = None
        for length in range(len(parts) - 1, 0, -1):
            try:
                target = import_module(".".join(parts[:length]))
            except ModuleNotFoundError as error:
                if error.name != ".".join(parts[:length]) and not ".".join(
                    parts[:length]
                ).startswith(f"{error.name}."):
                    raise ValueError(
                        f"Cannot import operation {self.call!r}: {error}"
                    ) from error
                continue
            try:
                for part in parts[length:]:
                    target = getattr(target, part)
            except AttributeError as error:
                raise ValueError(f"Cannot resolve operation {self.call!r}") from error
            break
        if target is None:
            raise ValueError(f"Cannot import operation {self.call!r}")
        return target

    def resolve(self, data: Any = None) -> Callable:
        """Resolve and validate a native bound method or imported function.

        Args:
            data: Native target required for method operations.

        Returns:
            Bound native method or custom callable, without executing it.
        """
        if self.method is not None:
            if data is None:
                raise ValueError("Resolving a method requires its native target")
            try:
                target = attrgetter(self.method)(data)
            except AttributeError as error:
                raise ValueError(f"Unknown native method {self.method!r}") from error
        else:
            target = self._import_call()
            from prefect.tasks import Task

            if isinstance(target, Task):
                target = target.fn
        name = self.method or self.call
        if (
            not callable(target)
            or inspect.isclass(target)
            or inspect.iscoroutinefunction(target)
            or inspect.isgeneratorfunction(target)
            or inspect.isasyncgenfunction(target)
            or inspect.iscoroutinefunction(getattr(target, "__call__", None))
            or inspect.isgeneratorfunction(getattr(target, "__call__", None))
            or inspect.isasyncgenfunction(getattr(target, "__call__", None))
        ):
            raise ValueError(f"Operation {name!r} must be a synchronous callable")
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError) as error:
            if self.method is not None:
                return target
            raise ValueError(f"Cannot inspect operation {name!r}: {error}") from error
        positional = () if self.method is not None else (object(),)
        try:
            signature.bind(*positional, **self.kwargs, **dict.fromkeys(self.inputs))
        except TypeError as error:
            raise ValueError(
                f"Invalid arguments for operation {name!r}: {error}"
            ) from error
        return target

    def apply(self, data: Any, *, inputs: Mapping[str, Any] | None = None) -> Any:
        """Invoke the operation on its native target with resolved named inputs."""
        arguments = dict(inputs) if inputs is not None else {}
        if arguments.keys() != self.inputs.keys():
            raise ValueError("Resolved operation inputs must match declared arguments")
        target = self.resolve(data)
        positional = () if self.method is not None else (data,)
        return target(*positional, **deepcopy(self.kwargs), **arguments)


class PreprocessingSpec(SpecModel):
    """Derive one named raster by selecting variables and applying operations.

    Args:
        raster: Source or recipe output used as the starting Dataset.
        variables: Optional variable names in their desired output order.
        operations: Native methods or custom callables, applied in order.
    """

    raster: RasterName
    variables: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    operations: tuple[OperationSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_variables(self) -> Self:
        if self.variables is not None:
            unique(self.variables, "recipe variables")
        return self
