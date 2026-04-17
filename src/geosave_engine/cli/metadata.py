from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def templates_root() -> Path:
    return package_root().parent / "templates"


def models_root() -> Path:
    return package_root() / "models"


def losses_root() -> Path:
    return package_root() / "losses"


def optimizers_root() -> Path:
    return package_root() / "optimizers"


@dataclass
class ModelInfo:
    class_name: str
    class_path: str
    task_map: dict[str, list[str]]
    docstring: str
    doc_link: str | None
    module_path: str


def normalize_slug(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").lower().split())


def resolve_from_choices(raw_value: str, choices: list[str]) -> str | None:
    normalized = normalize_slug(raw_value)
    norm_map = {normalize_slug(choice): choice for choice in choices}
    if normalized in norm_map:
        return norm_map[normalized]

    target_tokens = set(normalized.split())
    for choice in choices:
        choice_tokens = set(normalize_slug(choice).split())
        if target_tokens and target_tokens == choice_tokens:
            return choice
    return None
