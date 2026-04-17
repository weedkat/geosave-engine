from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlValidation:
    @staticmethod
    def validate(
        config_path: Path,
        *,
        required_mapping_keys: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Missing YAML file at {config_path}.")

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config_data: Any = yaml.safe_load(handle)
        except Exception as error:
            raise RuntimeError(
                f"Could not read YAML file '{config_path}': {error}"
            ) from error

        if config_data is None:
            raise ValueError(
                f"YAML file '{config_path}' is empty. Expected a mapping at root."
            )

        if not isinstance(config_data, dict):
            raise TypeError(
                f"YAML file '{config_path}' must contain a mapping at root."
            )

        for key in required_mapping_keys:
            if key not in config_data:
                raise KeyError(
                    f"YAML file '{config_path}' is missing required '{key}' section."
                )

            if not isinstance(config_data[key], dict):
                raise TypeError(
                    f"YAML key '{key}' in '{config_path}' must be a mapping."
                )

        return config_data
