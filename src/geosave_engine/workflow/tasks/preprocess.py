"""Execute ordered model preprocessing over native values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from inspect import signature
from typing import Any

from prefect import task

from geosave_engine.workflow.specs import ModelSpec, OperationSpec, Ref


class Preprocessor:
    """Compile and execute only a model specification's preprocessing calls."""

    def __init__(self, spec: ModelSpec) -> None:
        configuration = ModelSpec.model_validate(spec)
        self.operations: list[tuple[str, Callable[..., Any] | Ref, OperationSpec]] = []
        required: set[str] = set()
        assigned: set[str] = set()
        for name, operation in configuration.preprocessing.items():
            for reference in operation.references:
                if reference.root not in assigned:
                    required.add(reference.root)
            assigned.add(name)
            target = operation.call
            if isinstance(target, str):
                target = self.import_call(target)
                self.bind(target, operation.kwargs)
            self.operations.append((name, target, operation))
        self.required = frozenset(required)
        self.sources = {
            name: requirement
            for name, requirement in configuration.sources.items()
            if name in self.required
        }

    @staticmethod
    def import_call(path: str) -> Callable[..., Any]:
        """Import one validated module-level callable path."""
        module, name = path.rsplit(".", 1)
        target = getattr(import_module(module), name)
        if not callable(target):
            raise TypeError(f"Expected a callable, got {type(target).__name__}")
        return target

    @staticmethod
    def bind(target: Any, kwargs: Mapping[str, Any]) -> None:
        """Validate callability and inspectable keyword signatures."""
        if not callable(target):
            raise TypeError(f"Expected a callable, got {type(target).__name__}")
        try:
            parameters = signature(target)
        except (TypeError, ValueError):
            return
        parameters.bind(**kwargs)

    def validate_inputs(self, values: dict[str, Any]) -> None:
        """Validate every external reference and consumed source before work."""
        if missing := self.required - values.keys():
            raise ValueError(f"Missing preprocessing values: {sorted(missing)}")
        for name, requirement in self.sources.items():
            try:
                values[name] = requirement.select_raster(values[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Source {name!r}: {error}") from error

    def invoke(
        self,
        name: str,
        target: Callable[..., Any] | Ref,
        operation: OperationSpec,
        values: Mapping[str, Any],
    ) -> Any:
        """Resolve and invoke one call with stage context on failure."""
        try:
            function = target.resolve(values) if isinstance(target, Ref) else target
            kwargs = operation.resolve_kwargs(values)
            self.bind(function, kwargs)
            return function(**kwargs)
        except Exception as error:
            error.add_note(f"While executing preprocessing.{name}")
            raise

    def run(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        """Return fresh supplied bindings plus ordered preprocessing results."""
        values = dict(supplied)
        self.validate_inputs(values)
        for name, target, operation in self.operations:
            values[name] = self.invoke(name, target, operation, values)
        return values


@task(cache_policy=None, persist_result=False)
def preprocess(values: Mapping[str, Any], spec: ModelSpec) -> dict[str, Any]:
    """Execute model preprocessing without caching native lazy results."""
    return Preprocessor(spec).run(values)
