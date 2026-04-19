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
        raise AttributeError(f"Class '{class_name}' was not found in module '{module_path}'.")

    if not isinstance(cls, type):
        raise TypeError(
            f"Resolved object '{class_path}' is not a class (got type '{type(cls).__name__}')."
        )

    return cls


def instantiate_from_config(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    """Instantiate a regular class from a `{class_path, **kwargs}` mapping.

    Intended for raw torch classes or other non-GeoSave dependencies — calls
    the class constructor directly.
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
    """Instantiate a GeoSave builder via its `.build(...)` classmethod.

    Expected mapping: `{"class_path": "...", **build_kwargs}`. The resolved class
    must expose a callable `build()` method.
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


def instantiate_optimizers_from_config(
    optim_config: dict[str, Any] | list[dict[str, Any]],
    *,
    model: Any,
) -> tuple[dict[str, Any], ...]:
    """Instantiate Lightning optimizer configs from GeoSave YAML-style mappings.

    Each entry may carry a nested `scheduler` mapping; if the scheduler mapping
    itself contains a `config` key, those kwargs are merged into the returned
    Lightning `lr_scheduler` dict (frequency, monitor, interval, ...).
    """
    if isinstance(optim_config, dict):
        configs = [optim_config]
    elif isinstance(optim_config, list):
        configs = optim_config
    else:
        raise TypeError("optim_config must be a dictionary or a list of dictionaries")

    optimizer_configs: list[dict[str, Any]] = []

    for raw_optim_cfg in configs:
        if not isinstance(raw_optim_cfg, dict):
            raise ValueError("Each optimizer config entry must be a dictionary.")

        optim_cfg = dict(raw_optim_cfg)
        scheduler_cfg_raw = optim_cfg.pop("scheduler", None)

        class_path = optim_cfg.get("class_path")
        if isinstance(class_path, str) and class_path.startswith("geosave_engine."):
            optimizer = instantiate_from_config_build(optim_cfg, model=model)
        else:
            optimizer = instantiate_from_config(optim_cfg, model=model)
        optimizer_item: dict[str, Any] = {"optimizer": optimizer}

        if scheduler_cfg_raw is not None:
            if not isinstance(scheduler_cfg_raw, dict):
                raise ValueError("optimizer.scheduler must be a dictionary when provided.")

            scheduler_cfg = dict(scheduler_cfg_raw)
            lightning_scheduler_cfg = scheduler_cfg.pop("config", None)
            scheduler = instantiate_from_config(scheduler_cfg, optimizer=optimizer)

            if lightning_scheduler_cfg is None:
                optimizer_item["lr_scheduler"] = scheduler
            else:
                if not isinstance(lightning_scheduler_cfg, dict):
                    raise ValueError("optimizer.scheduler.config must be a dictionary.")
                optimizer_item["lr_scheduler"] = {
                    "scheduler": scheduler,
                    **dict(lightning_scheduler_cfg),
                }

        optimizer_configs.append(optimizer_item)

    return tuple(optimizer_configs)
