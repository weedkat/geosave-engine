from __future__ import annotations

from pathlib import Path

_EXCLUDED_TEMPLATE_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})


def templates_dir() -> Path:
    """Return bundled workspace template directory."""
    return Path(__file__).parents[2] / "templates"

def common_dir() -> Path:
    """Return bundled files copied into every workspace."""
    return templates_dir() / "common"

def task_dir() -> Path:
    """Return bundled task template directory."""
    return templates_dir() / "tasks"

def boilerplate_dir() -> Path:
    """Return bundled component template directory."""
    return templates_dir() / "boilerplate"

def get_template(root: Path, include_file: bool = False) -> dict[str, list[str]]:
    """Return a dictionary of tasks and their methods."""
    templates = {}
    for path in root.iterdir():
        if path.is_dir() and path.name not in _EXCLUDED_TEMPLATE_NAMES:
            for item in path.iterdir():
                if item.name in _EXCLUDED_TEMPLATE_NAMES:
                    continue
                if item.is_dir():
                    templates.setdefault(path.name, []).append(item.name)
                elif include_file and item.is_file():
                    templates.setdefault(path.name, []).append(item.name)
                    
    return templates

def get_tasks() -> dict[str, list[str]]:
    """Return a dictionary of tasks and their methods."""
    return get_template(task_dir())

def get_boilerplate() -> dict[str, list[str]]:
    """Return a dictionary of boilerplate and their files."""
    return get_template(boilerplate_dir(), include_file=True)


if __name__ == "__main__":
    print(get_tasks())
    print(get_boilerplate())