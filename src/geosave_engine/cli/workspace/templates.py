from __future__ import annotations

from pathlib import Path

_EXCLUDED_TEMPLATE_NAMES = frozenset({"__pycache__", "common", ".ipynb_checkpoints"})


def templates_dir() -> Path:
    """Return bundled workspace template directory."""
    return Path(__file__).parents[2] / "templates"


def common_template_dir() -> Path:
    """Return bundled files copied into every workspace."""
    return templates_dir() / "common"


def get_task_templates() -> list[Path]:
    """Return available task template directories."""
    return [
        path
        for path in templates_dir().iterdir()
        if path.is_dir() and path.name not in _EXCLUDED_TEMPLATE_NAMES
    ]

def get_method_templates() -> dict[str, dict[str, Path]]:
    """Map task names to their method template directories."""
    methods: dict[str, dict[str, Path]] = {}
    for task in get_task_templates():
        task_methods = {
            path.name: path
            for path in task.iterdir()
            if path.is_dir() and path.name not in _EXCLUDED_TEMPLATE_NAMES
        }
        if task_methods:
            methods[task.name] = task_methods
    return methods
