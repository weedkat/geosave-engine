from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from geosave_engine.utils import filter_kwargs

if TYPE_CHECKING:
    from geosave_engine.ml.models.contract import ContextChain

MODEL_REGISTRY: dict[str, dict[str, type[nn.Module]]] = {}


def register_model(stage: str, name: str):
    """Register an nn.Module class under a chain stage, for ``build_model``.

    Args:
        stage: Stage this class can fill, e.g. ``"encoder"``, ``"decoder"``,
            ``"head"``, ``"model"`` (monolith — a single-stage chain).
        name: Registry key within that stage, e.g. ``"dinov3"``.

    Returns:
        The decorator — registers ``cls`` under ``MODEL_REGISTRY[stage][name.upper()]``
        and returns it unchanged.

    Raises:
        ValueError: ``stage``/``name`` (case-insensitive) is already registered
            to a different class — two files silently overwriting each
            other's entry is worse than failing at import time.

    Examples:
        >>> @register_model('encoder', 'dinov3')
        ... class DINOv3(nn.Module): ...
    """
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        key = name.upper()
        existing = MODEL_REGISTRY.get(stage, {}).get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"'{stage}'/'{name}' already registered to {existing.__name__}, "
                f"can't also register {cls.__name__} — pick a different name."
            )
        MODEL_REGISTRY.setdefault(stage, {})[key] = cls
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
    """Look for ``'{stage}_{attr}'`` pattern on cls.__init__ parameters, 
        and if ``built[stage]`` exists, pull that attribute's value.

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
        ``ContextChain`` wrapping the built stages, keyed by stage name. The
        first declared stage gets ``is_entry = True`` set on its instance —
        ``stages``' declared order already has to be correct for the
        auto-wiring above, so the entry is exactly whichever stage is first.

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

    first_stage = next(iter(built))
    setattr(built[first_stage], 'is_entry', True)
    return ContextChain(**built)
