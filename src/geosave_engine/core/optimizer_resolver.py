from __future__ import annotations

from typing import Any

from geosave_engine.core.class_resolver import (
    instantiate_from_config,
    instantiate_from_config_build,
)


def instantiate_optimizers_from_config(
    optim_config: dict[str, Any] | list[dict[str, Any]],
    *,
    model: Any,
) -> tuple[dict[str, Any], ...]:
    """Instantiate Lightning optimizer configs from GeoSave YAML-style mappings."""
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
