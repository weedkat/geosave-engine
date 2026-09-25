from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, TypedDict


class BuildSpec(TypedDict, total=False):
    """Describe construction by registered name or importable class."""

    name: str
    class_path: str
    init_args: dict[str, Any]


def resolve[T](
    spec: BuildSpec,
    registry: Mapping[str, Callable[..., T]],
    base: type[T],
) -> Callable[..., T]:
    """Resolve exactly one configured selector to a compatible factory.

    Args:
        spec: Registered name or class path with optional constructor arguments.
        registry: Available named factories.
        base: Required base class for imported classes.

    Returns:
        Selected registered factory or imported class.

    Raises:
        ValueError: If fields or selectors are invalid.
        TypeError: If arguments or the imported class have the wrong type.
    """
    unknown = set(spec) - {"name", "class_path", "init_args"}
    if unknown:
        raise ValueError(f"Unknown construction fields: {sorted(unknown)}; use init_args")
    if ("name" in spec) == ("class_path" in spec):
        raise ValueError("Specify exactly one of name or class_path")
    if not isinstance(spec.get("init_args", {}), dict):
        raise TypeError("init_args must be a dict of constructor arguments")

    if "name" in spec:
        name = spec["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty registered factory name")
        factories = {key.casefold(): factory for key, factory in registry.items()}
        try:
            return factories[name.casefold()]
        except KeyError:
            raise ValueError(
                f"Unknown name {name!r}; available: {list(registry)}"
            ) from None

    path = spec.get("class_path")
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"class_path must include a module and class: {path!r}")
    module_name, _, attribute = path.rpartition(".")
    factory = getattr(import_module(module_name), attribute, None)
    if not isinstance(factory, type) or not issubclass(factory, base):
        raise TypeError(f"{path!r} must name a {base.__name__} subclass")
    return factory
