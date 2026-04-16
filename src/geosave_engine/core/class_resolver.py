from __future__ import annotations

from importlib import import_module
from typing import Any


def resolve_class(class_path: str) -> type[Any]:
    """Resolve a dotted class path such as 'torch.optim.AdamW'."""
    if not class_path or "." not in class_path:
        raise ValueError(
            f"Invalid class_path '{class_path}'. Expected dotted path like 'torch.optim.AdamW'."
        )

    module_path, class_name = class_path.rsplit(".", 1)

    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import module '{module_path}' from class_path '{class_path}'."
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Class '{class_name}' was not found in module '{module_path}'."
        )

    if not isinstance(cls, type):
        raise TypeError(
            f"Resolved object '{class_path}' is not a class (got type '{type(cls).__name__}')."
        )

    return cls


def instantiate_from_config(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    """Instantiate a regular class from a config mapping.

    This uses the class constructor directly and is intended for raw Torch
    classes or other non-GeoSave dependencies.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    kwargs = dict(config)
    class_path = kwargs.pop("class_path", None)
    if not isinstance(class_path, str):
        raise ValueError("config must include 'class_path' as a string")

    cls = resolve_class(class_path)
    return cls(**kwargs, **extra_kwargs)


def instantiate_from_config_build(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    """
    Instantiate a GeoSave builder from a config mapping.

    Expected pattern:
    {
        "class_path": "some.module.ClassName",
        "arg1": value1,
        "arg2": value2,
    }

    The function pops `class_path` from a shallow copy of `config` and passes the
    remaining keys as keyword arguments to the resolved class `build` method.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    kwargs = dict(config)
    class_path = kwargs.pop("class_path", None)
    if not isinstance(class_path, str):
        raise ValueError("config must include 'class_path' as a string")

    cls = resolve_class(class_path)
    build = getattr(cls, "build", None)
    if build is None or not callable(build):
        raise TypeError(
            f"GeoSave builder '{class_path}' must define a callable build() method."
        )

    return build(**kwargs, **extra_kwargs)


