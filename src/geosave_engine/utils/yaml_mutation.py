from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from geosave_engine.utils.yaml_validation import YamlValidation


class YamlMutation:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config_data = YamlValidation.validate(config_path)

    def inject(
        self,
        key: str | tuple[str, ...],
        value: Any,
        *,
        create_missing_parents: bool = False,
        required_mapping_keys: tuple[str, ...] = (),
    ) -> bool:
        key_path = self._normalize_key_path(key)
        if not key_path:
            raise ValueError("key must contain at least one path segment.")

        for required_key in required_mapping_keys:
            current = self.config_data.get(required_key)
            if current is None:
                raise KeyError(
                    f"Missing required key '{required_key}' in '{self.config_path}'."
                )
            if not isinstance(current, dict):
                raise TypeError(
                    f"Required key '{required_key}' in '{self.config_path}' must be a mapping."
                )

        parent = self.config_data
        for key_part in key_path[:-1]:
            current = parent.get(key_part)

            if current is None:
                if not create_missing_parents:
                    raise KeyError(
                        f"Missing key '{key_part}' while resolving path {'.'.join(key_path)} in '{self.config_path}'."
                    )
                parent[key_part] = {}
                current = parent[key_part]

            if not isinstance(current, dict):
                raise TypeError(
                    f"Key '{key_part}' in '{self.config_path}' must be a mapping to set {'.'.join(key_path)}."
                )

            parent = current

        parent[key_path[-1]] = value
        self._save_yaml()
        return True

    def _normalize_key_path(self, key: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(key, tuple):
            return key
        if not isinstance(key, str):
            raise TypeError("key must be a dot path string or tuple[str, ...].")
        return tuple(part.strip() for part in key.split(".") if part.strip())

    def _save_yaml(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(self.config_data, handle, sort_keys=False)
        except Exception as error:
            raise RuntimeError(
                f"Could not write config file '{self.config_path}': {error}"
            ) from error
