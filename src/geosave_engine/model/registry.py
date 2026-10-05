from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import torch.nn as nn

from geosave_engine.model.chain import ModelChain
from geosave_engine.model.chain.published import published_kwargs
from geosave_engine.model.factory import BuildSpec

MODEL_REGISTRY: dict[str, dict[str, Callable[..., nn.Module]]] = {}


def register_model[Factory: Callable[..., nn.Module]](
    stage: str, name: str
) -> Callable[[Factory], Factory]:
    """Register a model factory under a stage and name.

    Args:
        stage: Stage the factory fills, such as encoder, decoder, head, or model.
        name: Case-insensitive name used in construction specifications.

    Returns:
        Decorator that registers and returns the class or factory unchanged.

    Raises:
        ValueError: The stage and name already identify another factory.
    """

    def decorator(factory: Factory) -> Factory:
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
    from geosave_engine.model import decoder, encoder, head, monolith  # noqa: F401

    if stage is not None:
        return {stage: list(MODEL_REGISTRY.get(stage, {}))}
    return {name: list(factories) for name, factories in MODEL_REGISTRY.items()}


def build_model(
    stages: Mapping[str, BuildSpec | Mapping[str, Any]],
) -> ModelChain:
    """Construct a reproducible model chain from ordered stage specifications.

    Args:
        stages: Ordered registered names or importable module classes.

    Returns:
        Model chain with an independent resolved construction recipe.
    """
    from geosave_engine.model import decoder, encoder, head, monolith  # noqa: F401

    if not stages:
        raise ValueError("Supply at least one model stage")

    modules: dict[str, nn.Module] = {}
    recipe: dict[str, dict[str, Any]] = {}
    for stage, value in stages.items():
        spec = value if isinstance(value, BuildSpec) else BuildSpec.model_validate(value)
        factory = spec.resolve(MODEL_REGISTRY.get(stage, {}), nn.Module)
        try:
            kwargs = published_kwargs(factory, modules, spec.init_args)
            signature = inspect.signature(factory)
            bound = signature.bind(**kwargs)
            bound.apply_defaults()
            init_args: dict[str, object] = {}
            for name, argument in bound.arguments.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    init_args.update(argument)
                elif kind is not inspect.Parameter.VAR_POSITIONAL:
                    init_args[name] = argument
            init_args = deepcopy(init_args)
            modules[stage] = factory(**kwargs)
        except TypeError as error:
            raise TypeError(f"Building stage {stage!r} failed: {error}") from error

        if "pretrained" in signature.parameters:
            init_args["pretrained"] = False
        recipe[stage] = {
            **spec.model_dump(exclude_none=True),
            "init_args": init_args,
        }

    model = ModelChain(**modules)
    model._stage_specs = deepcopy(recipe)
    return model
