from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
)
from torch.nn import CrossEntropyLoss

from geosave_engine.ml.loss import ProbOhemCrossEntropy2d
from geosave_engine.utils import filter_kwargs

if TYPE_CHECKING:
    from geosave_engine.ml.models.contract import ContextChain

import geosave_engine.ml.optimizer.adamw as adamw
import geosave_engine.ml.optimizer.adam as adam
import geosave_engine.ml.optimizer.sgd as sgd
import geosave_engine.ml.optimizer.rmsprop as rmsprop
import geosave_engine.ml.optimizer.adagrad as adagrad


LOSSES = {
    'CELoss': CrossEntropyLoss,
    'OHEMLoss': ProbOhemCrossEntropy2d,
}

OPTIMIZERS = {
    'AdamW': adamw,
    'Adam': adam,
    'SGD': sgd,
    'RMSprop': rmsprop,
    'Adagrad': adagrad,
}

SCHEDULERS = {
    "LRScheduler": LRScheduler,
    "CosineAnnealingLR": CosineAnnealingLR,
}

MODEL_REGISTRY: dict[str, dict[str, type[nn.Module]]] = {}

def uppercase_keys(d: dict) -> dict:
    """Return copy of ``d`` with all keys uppercased."""
    return {k.upper(): v for k, v in d.items()}


def builder(name: str, config: dict, registry: dict):
    """Instantiate entry from registry by name.

    Case-insensitive. Use for losses and models.

    Args:
        name: Registry key (e.g. ``"CELoss"``).
        config: Keyword args passed to the constructor.
        registry: Mapping of name → callable.

    Returns:
        Instantiated object from registry.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    reg = uppercase_keys(registry)
    key = name.upper()
    if key not in reg:
        raise ValueError(f"Unknown '{name}'. Available: {list(registry.keys())}")
    return reg[key](**config)


def method_builder(name: str, config: dict, registry: dict):
    """Instantiate via dot-notation method dispatch (``"key.method"``).

    Use for optimizers where variant is a module-level function.
    If no method given, falls back to ``default``.

    Args:
        name: ``"key"`` or ``"key.method"`` (e.g. ``"AdamW.split"``).
        config: Keyword args passed to the method.
        registry: Mapping of name → module with callable methods.

    Returns:
        Result of calling ``registry[key].method(**config)``.

    Raises:
        ValueError: If key or method not found in registry.
    """
    if "." not in name:
        name = name + ".default"

    reg = uppercase_keys(registry)
    raw_key, method = name.split(".", 1)
    key = raw_key.upper()
    if key not in reg:
        raise ValueError(f"Unknown key '{raw_key}'. Available: {list(registry.keys())}")

    entry = reg[key]
    if not hasattr(entry, method):
        available = [m for m in dir(entry) if not m.startswith("_")]
        raise ValueError(f"Unknown method '{method}' on '{raw_key}'. Available: {available}")

    return getattr(entry, method)(**config)


def register_model(stage: str, name: str):
    """Register an nn.Module class under a chain stage, for ``build_model``.

    Args:
        stage: Stage this class can fill, e.g. ``"encoder"``, ``"decoder"``,
            ``"head"``, ``"model"`` (monolith — a single-stage chain).
        name: Registry key within that stage, e.g. ``"dinov3"``.

    Returns:
        The decorator — registers ``cls`` under ``MODEL_REGISTRY[stage][name.upper()]``
        and returns it unchanged.

    Examples:
        >>> @register_model('encoder', 'dinov3')
        ... class DINOv3(nn.Module): ...
    """
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        MODEL_REGISTRY.setdefault(stage, {})[name.upper()] = cls
        return cls
    return decorator


def _resolve_stage_cls(stage: str, spec: str | type[nn.Module]) -> type[nn.Module]:
    """Resolve a stage's class — a registry key, or an already-given class.

    Args:
        stage: Stage name, e.g. ``"encoder"`` — looked up in ``MODEL_REGISTRY``.
        spec: Registry key (case-insensitive) or an ``nn.Module`` subclass directly.

    Returns:
        The resolved ``nn.Module`` subclass.

    Raises:
        ValueError: If ``spec`` is a string not found in ``MODEL_REGISTRY[stage]``.
    """
    if not isinstance(spec, str):
        return spec
    
    import geosave_engine.ml.models  # noqa: F401 -- populates MODEL_REGISTRY via @register_model side effects

    available = MODEL_REGISTRY.get(stage, {})
    key = spec.upper()
    if key not in available:
        raise ValueError(f"Unknown {stage} '{spec}'. Available: {list(available)}")
    return available[key]


def _stage_kwargs(cls: type[nn.Module], built: dict[str, nn.Module]) -> dict[str, Any]:
    """Auto-wire constructor params shaped ``'{stage}_{attr}'`` from earlier-built stages.

    For every ``__init__`` param, split on the first underscore. If the
    first part names an already-built stage, resolve the rest as an
    attribute on that stage's instance — e.g. ``encoder_out_channels`` reads
    ``built['encoder'].out_channels``. The attribute name must match how the
    producing stage names it (its own ``out_channels``, not the consumer's
    concept for the same value) — get it wrong and this raises immediately,
    it never silently no-ops. Not limited to the immediately preceding
    stage — any already-built stage is addressable this way, which is what
    lets a fan-in stage pull from two independent earlier branches.

    Args:
        cls: Class about to be constructed for the current stage.
        built: Already-built stages, keyed by stage name.

    Returns:
        Param name to resolved value, for whichever params matched.

    Raises:
        AttributeError: A param's prefix matches a built stage, but that
            stage's instance has no such attribute.
    """
    resolved: dict[str, Any] = {}
    for param in inspect.signature(cls.__init__).parameters:
        stage, _, attr = param.partition('_')
        if stage not in built or not attr:
            continue
        if not hasattr(built[stage], attr):
            raise AttributeError(
                f"{cls.__name__}.__init__ param '{param}' implies "
                f"{type(built[stage]).__name__}.{attr}, but no such attribute exists"
            )
        resolved[param] = getattr(built[stage], attr)
    return resolved


def build_model(
    stages: dict[str, str | type[nn.Module]],
    config: dict[str, dict[str, Any]] | None = None,
) -> ContextChain:
    """Build a ContextChain by resolving, wiring, and constructing each stage in order.

    No topology knowledge lives here — ``stages`` says what to build and in
    what sequence (dict insertion order is build order); per-stage params
    come from auto-wiring (see ``_stage_kwargs``) plus whatever the caller
    places in ``config[stage]``. Nothing is broadcast across stages — a
    value only reaches a stage if it's auto-wired from an earlier stage's
    attribute or explicitly placed under that stage's own ``config`` entry,
    so same-named params on different stages (e.g. two classes both taking
    ``in_channels``) never collide. A monolith is just a single-entry
    ``stages`` dict — no separate code path.

    Args:
        stages: Stage name to registry key (or class directly), in build
            order, e.g. ``{'encoder': 'dinov3', 'decoder': 'dpt', 'head': 'dense'}``
            or ``{'model': 'granite_geospatial_biomass'}``.
        config: Stage name to that stage's own constructor kwargs.

    Returns:
        ``ContextChain`` wrapping the built stages, keyed by stage name.

    Raises:
        TypeError: A stage's constructor rejects the kwargs assembled for it
            (missing/invalid arg) — re-raised with the stage name attached.
    """
    from geosave_engine.ml.models.contract import ContextChain

    config = config or {}
    built: dict[str, nn.Module] = {}
    for stage in stages:
        cls = _resolve_stage_cls(stage, stages[stage])
        kwargs = {**_stage_kwargs(cls, built), **config.get(stage, {})}
        try:
            built[stage] = cls(**filter_kwargs(cls, kwargs))
        except TypeError as e:
            raise TypeError(f"building '{stage}' ({cls.__name__}) failed: {e}") from e
    return ContextChain(built)


def build_loss(name: str, config: dict, registry: dict = LOSSES) -> nn.Module:
    """Build loss by name from registry.

    Args:
        name: Registry key (e.g. ``"CELoss"``).
        config: Keyword args passed to the constructor.
        registry: Mapping of name → loss class. Defaults to ``LOSSES``.

    Returns:
        Instantiated loss ``nn.Module``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return builder(name, config, registry)


def build_optimizer(name: str, model: nn.Module, config: dict, registry: dict = OPTIMIZERS) -> Optimizer:
    """Build optimizer by name from registry.

    Args:
        name: Registry key; supports dot notation (e.g. ``"AdamW.split"``).
        config: Keyword args passed to the optimizer constructor.
        model: Model whose parameters are passed to the optimizer.
        registry: Mapping of name → optimizer module. Defaults to ``OPTIMIZERS``.

    Returns:
        Instantiated ``Optimizer``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return method_builder(name, {**config, "model": model}, registry)


def build_scheduler(name: str, optimizer: Optimizer, config: dict, registry: dict = SCHEDULERS) -> LRScheduler | None:
    """Build LR scheduler by name from registry.

    Args:
        name: Registry key (e.g. ``"CosineAnnealingLR"``).
        optimizer: Optimizer passed to scheduler constructor.
        config: Keyword args passed to the scheduler constructor.
        registry: Mapping of name → scheduler class. Defaults to ``SCHEDULERS``.

    Returns:
        Instantiated ``LRScheduler``, or ``None`` if ``name`` is ``None``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return builder(name, {**config, "optimizer": optimizer}, registry)
