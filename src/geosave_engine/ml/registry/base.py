from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, TypedDict


class BuildSpec(TypedDict, total=False):
    """Describe construction through exactly one registered name or class path.

    Args:
        name: Registered factory name, matched without case sensitivity.
        class_path: Dotted import path to a class.
        init_args: Keyword arguments passed to the factory or class.
    """

    name: str
    class_path: str
    init_args: dict[str, Any]


def resolve[T](
    spec: BuildSpec,
    registry: Mapping[str, Callable[..., T]],
    base: type[T],
) -> Callable[..., T]:
    """Resolve a construction specification to a factory or imported class.

    Args:
        spec: Exactly one selector and optional constructor arguments.
        registry: Named factories producing instances of the expected type.
        base: Required base class for an imported class.

    Returns:
        Factory selected by name or class path.

    Raises:
        ValueError: Selectors, specification fields, or the name are invalid.
        TypeError: Arguments are not a dict or the imported class has the wrong type.
        ImportError: The class path cannot be imported.
    """
    unknown = set(spec) - {"name", "class_path", "init_args"}
    if unknown:
        raise ValueError(
            f"Unknown construction fields: {sorted(unknown)}; use init_args"
        )
    if ("name" in spec) == ("class_path" in spec):
        raise ValueError("Specify exactly one of name or class_path")
    if not isinstance(spec.get("init_args", {}), dict):
        raise TypeError("init_args must be a dict of constructor arguments")

    if "name" in spec:
        name = spec["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty registered factory name")
        factories = {key.casefold(): factory for key, factory in registry.items()}
        if name.casefold() not in factories:
            raise ValueError(f"Unknown name {name!r}; available: {list(registry)}")
        return factories[name.casefold()]

    path = spec.get("class_path")
    if not isinstance(path, str) or not path:
        raise ValueError("class_path must be a non-empty dotted class path")
    module, separator, attribute = path.rpartition(".")
    if not separator or not module or not attribute:
        raise ValueError(f"class_path must include a module and class: {path!r}")
    cls = getattr(import_module(module), attribute, None)
    if not isinstance(cls, type) or not issubclass(cls, base):
        raise TypeError(f"{path!r} must name a {base.__name__} subclass")
    return cls
