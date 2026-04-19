from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(
    config_path: Path,
    *,
    required_mapping_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load and validate a YAML mapping file.

    Ensures the document exists, is a mapping at the root, and contains every
    key in `required_mapping_keys` as a nested mapping.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Missing YAML file at {config_path}.")

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data: Any = yaml.safe_load(handle)
    except Exception as error:
        raise RuntimeError(f"Could not read YAML file '{config_path}': {error}") from error

    if data is None:
        raise ValueError(f"YAML file '{config_path}' is empty. Expected a mapping at root.")

    if not isinstance(data, dict):
        raise TypeError(f"YAML file '{config_path}' must contain a mapping at root.")

    for key in required_mapping_keys:
        if key not in data:
            raise KeyError(f"YAML file '{config_path}' is missing required '{key}' section.")
        if not isinstance(data[key], dict):
            raise TypeError(f"YAML key '{key}' in '{config_path}' must be a mapping.")

    return data


def save_yaml(config_path: Path, data: dict[str, Any]) -> None:
    """Write `data` back to `config_path` preserving key order."""
    try:
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    except Exception as error:
        raise RuntimeError(f"Could not write YAML file '{config_path}': {error}") from error


def _normalize_key_path(key: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(key, tuple):
        return key
    if not isinstance(key, str):
        raise TypeError("key must be a dot-path string or tuple[str, ...].")
    return tuple(part.strip() for part in key.split(".") if part.strip())


def inject_value(
    data: dict[str, Any],
    key: str | tuple[str, ...],
    value: Any,
    *,
    create_missing_parents: bool = False,
    required_mapping_keys: tuple[str, ...] = (),
) -> None:
    """Set `data[key_path] = value` in place, walking a dotted path.

    Raises KeyError if a parent is missing and `create_missing_parents=False`,
    and TypeError if any path segment resolves to a non-mapping.
    """
    key_path = _normalize_key_path(key)
    if not key_path:
        raise ValueError("key must contain at least one path segment.")

    for required_key in required_mapping_keys:
        current = data.get(required_key)
        if current is None:
            raise KeyError(f"Missing required key '{required_key}'.")
        if not isinstance(current, dict):
            raise TypeError(f"Required key '{required_key}' must be a mapping.")

    parent = data
    for segment in key_path[:-1]:
        current = parent.get(segment)
        if current is None:
            if not create_missing_parents:
                raise KeyError(
                    f"Missing key '{segment}' while resolving path {'.'.join(key_path)}."
                )
            parent[segment] = {}
            current = parent[segment]
        if not isinstance(current, dict):
            raise TypeError(
                f"Key '{segment}' must be a mapping to set {'.'.join(key_path)}."
            )
        parent = current

    parent[key_path[-1]] = value


def inject_into_file(
    config_path: Path,
    key: str | tuple[str, ...],
    value: Any,
    *,
    create_missing_parents: bool = False,
    required_mapping_keys: tuple[str, ...] = (),
) -> None:
    """Load, mutate, and save a YAML file in one shot."""
    data = load_yaml(config_path)
    inject_value(
        data,
        key,
        value,
        create_missing_parents=create_missing_parents,
        required_mapping_keys=required_mapping_keys,
    )
    save_yaml(config_path, data)
