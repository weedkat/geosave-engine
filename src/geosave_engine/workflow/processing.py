"""Independently loadable model processing stages over native Python objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Literal, Self

from .spec import ModelSpec, Ref
from .spec.references import references, resolve


def _bind(target: Any, kwargs: Mapping[str, Any]) -> None:
    """Check callability and inspectable keyword signatures without executing."""
    if not callable(target):
        raise TypeError(f"Expected a callable, got {type(target).__name__}")
    try:
        parameters = signature(target)
    except (TypeError, ValueError):
        # Some native callables, including dict, expose no Python signature.
        return
    parameters.bind(**kwargs)


class Processor:
    """Execute one model-spec stage over supplied native Python values.

    Args:
        spec: Model document; inactive stages are never imported or executed.
        stage: `preprocessing` or `postprocessing`.

    Examples:
        >>> prepare = Processor.load("model_spec.yaml", stage="preprocessing")
        >>> prepared = prepare({"optical": optical})
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        stage: Literal["preprocessing", "postprocessing"],
    ) -> None:
        if stage not in ("preprocessing", "postprocessing"):
            raise ValueError(f"Unsupported processing stage: {stage!r}")
        # Keep this processor independent of later edits to the spec.
        configuration = spec.validated_copy()
        self._stage = stage
        self._operations: list[
            tuple[str, Callable[..., Any] | Ref, dict[str, Any]]
        ] = []
        self._required: set[str] = set()
        assigned: set[str] = set()
        # Collect the selected stage's calls and required input names.
        for name, operation in getattr(configuration, stage).items():
            for reference in references([operation.call, operation.kwargs]):
                if reference.root not in assigned:
                    self._required.add(reference.root)
            assigned.add(name)
            target = operation.call
            if isinstance(target, str):
                module, attribute = target.rsplit(".", 1)
                target = getattr(import_module(module), attribute)
                _bind(target, operation.kwargs)
            self._operations.append((name, target, operation.kwargs))
        # Validate only raster sources consumed by this stage.
        self._sources = {
            name: requirement
            for name, requirement in configuration.sources.items()
            if name in self._required
        }

    @classmethod
    def load(
        cls, path: str | Path, *, stage: Literal["preprocessing", "postprocessing"]
    ) -> Self:
        """Load this stage from a YAML file or local model artifact directory.

        Args:
            path: YAML filename or directory containing `model_spec.yaml`.
            stage: `preprocessing` or `postprocessing`.

        Returns:
            Configured processor; inactive stages' callables are not imported.
        """
        return cls(ModelSpec.load(path), stage=stage)

    def __call__(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        """Call the stage with supplied values and return their updated bindings.

        References resolve before each result is assigned. Rebinding does not
        mutate the previous value; explicitly invoked methods may mutate their
        receivers. Literal argument containers are fresh on every invocation.

        Args:
            supplied: Native data and callables required by the active stage.

        Returns:
            Fresh dictionary of supplied values plus this stage's named results.

        Raises:
            ValueError: A required name or source contract is unsatisfied.
            TypeError: A target is not callable or its arguments are invalid.
            AttributeError: A referenced attribute is absent.
        """
        # Check all required inputs before running any operation.
        values = dict(supplied)
        if missing := self._required - values.keys():
            raise ValueError(f"Missing {self._stage} values: {sorted(missing)}")
        for name, requirement in self._sources.items():
            try:
                values[name] = requirement.select_raster(values[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Source {name!r}: {error}") from error
        # Run each call and store its output under the declared name.
        for name, target, arguments in self._operations:
            try:
                function = target.resolve(values) if isinstance(target, Ref) else target
                kwargs = resolve(arguments, values)
                _bind(function, kwargs)
                values[name] = function(**kwargs)
            except Exception as error:
                error.add_note(f"While executing {self._stage}.{name}")
                raise
        return values
