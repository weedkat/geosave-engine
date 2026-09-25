from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import Any, TypeVar, TypedDict

import torch.nn as nn

type ModelFactory = Callable[..., nn.Module]
_Factory = TypeVar("_Factory", bound=ModelFactory)
MODEL_REGISTRY: dict[str, dict[str, ModelFactory]] = {}


class StageSpec(TypedDict, total=False):
    """Describe one model stage selected by registered name or class path.

    Args:
        name: Registered factory name, matched without case sensitivity.
        class_path: Dotted import path to an ``nn.Module`` subclass.
        init_args: Keyword arguments passed to the selected stage constructor.
    """

    name: str
    class_path: str
    init_args: dict[str, Any]


def register_model(stage: str, name: str) -> Callable[[_Factory], _Factory]:
    """Register a model factory under a stage and name.

    Args:
        stage: Stage the factory fills, such as encoder, decoder, head, or model.
        name: Case-insensitive name used in construction specifications.

    Returns:
        Decorator that registers and returns the class or factory unchanged.

    Raises:
        ValueError: The stage and name already identify another factory.
    """

    def decorator(factory: _Factory) -> _Factory:
        key = name.upper()
        existing = MODEL_REGISTRY.get(stage, {}).get(key)
        if existing is not None and existing is not factory:
            raise ValueError(f"{stage!r}/{name!r} already identifies another factory")
        MODEL_REGISTRY.setdefault(stage, {})[key] = factory
        return factory

    return decorator


def list_models(stage: str | None = None) -> dict[str, list[str]]:
    """List registered model names by stage.

    Args:
        stage: Stage to list. None lists every stage.

    Returns:
        Stage names mapped to registered names; an unknown stage has an empty list.
    """
    import geosave_engine.ml.models  # noqa: F401

    if stage is not None:
        return {stage: list(MODEL_REGISTRY.get(stage, {}))}
    return {name: list(factories) for name, factories in MODEL_REGISTRY.items()}


@dataclass(frozen=True)
class BuiltStages:
    """Constructed stages and their reproducible initialization arguments.

    Args:
        modules: Stage names mapped to module instances in construction order.
        config: Construction specifications with defaults and Published values.
    """

    modules: dict[str, nn.Module]
    config: dict[str, StageSpec]


def _resolve_stage(
    spec: StageSpec,
    factories: Mapping[str, ModelFactory],
) -> ModelFactory:
    """Resolve a strict model stage selector to its factory."""
    unknown = set(spec) - {"name", "class_path", "init_args"}
    if unknown:
        raise ValueError(f"Unknown stage fields: {sorted(unknown)}; use init_args")
    if ("name" in spec) == ("class_path" in spec):
        raise ValueError("Specify exactly one of name or class_path")
    if not isinstance(spec.get("init_args", {}), dict):
        raise TypeError("init_args must be a dict of constructor arguments")
    if "name" in spec:
        name = spec["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty registered factory name")
        matched = {name.casefold(): factory for name, factory in factories.items()}
        if name.casefold() not in matched:
            raise ValueError(f"Unknown model stage {name!r}; available: {list(factories)}")
        return matched[name.casefold()]
    path = spec["class_path"]
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"class_path must include a module and class: {path!r}")
    module, _, attribute = path.rpartition(".")
    stage = getattr(import_module(module), attribute)
    if not isinstance(stage, type) or not issubclass(stage, nn.Module):
        raise TypeError(f"{path!r} must name an nn.Module subclass")
    return stage


def build_stages(stages: dict[str, StageSpec]) -> BuiltStages:
    """Construct stages and record the arguments needed to reconstruct them.

    Args:
        stages: Ordered factory or class specifications. Explicit arguments
            override attributes published by earlier stages.

    Returns:
        Modules and independent construction specifications with defaults filled.
        Saved pretrained arguments are False; exported weights supply the state.

    Raises:
        ValueError: Stages are empty or a construction specification is invalid.
        TypeError: Constructor arguments or Published attributes are incompatible.
    """
    from geosave_engine.ml.models.contract.published import published_kwargs

    if not stages:
        raise ValueError("Supply at least one model stage")
    modules: dict[str, nn.Module] = {}
    config: dict[str, StageSpec] = {}
    for stage, spec in stages.items():
        factory = _resolve_stage(spec, MODEL_REGISTRY.get(stage, {}))
        try:
            kwargs = published_kwargs(factory, modules, spec.get("init_args", {}))
            signature = inspect.signature(factory)
            bound = signature.bind(**kwargs)
            bound.apply_defaults()
            init_args: dict[str, object] = {}
            for name, value in bound.arguments.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    init_args.update(value)
                elif kind is not inspect.Parameter.VAR_POSITIONAL:
                    init_args[name] = value
            # Snapshot before construction: factories may mutate mutable arguments.
            init_args = deepcopy(init_args)
            modules[stage] = factory(**kwargs)
        except TypeError as error:
            raise TypeError(f"Building stage {stage!r} failed: {error}") from error
        if "pretrained" in signature.parameters:
            init_args["pretrained"] = False
        config[stage] = {**spec, "init_args": init_args}
    return BuiltStages(modules=modules, config=config)
