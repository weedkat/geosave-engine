from __future__ import annotations

from pathlib import Path

_EXCLUDED_TEMPLATE_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})

TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
COMMON_DIR = TEMPLATES_DIR / "common"
TASK_DIR = TEMPLATES_DIR / "tasks"
BOILERPLATE_DIR = TEMPLATES_DIR / "boilerplate"


def _list_templates(root: Path, include_file: bool = False) -> dict[str, list[str]]:
    """Map each template directory under `root` to the entries it offers.

    Args:
        root: Directory holding one subdirectory per template.
        include_file: Also list files, not only subdirectories.

    Returns:
        Template directory names mapped to their entry names.
    """
    templates: dict[str, list[str]] = {}
    for path in root.iterdir():
        if not path.is_dir() or path.name in _EXCLUDED_TEMPLATE_NAMES:
            continue
        for item in path.iterdir():
            if item.name in _EXCLUDED_TEMPLATE_NAMES:
                continue
            if item.is_dir() or (include_file and item.is_file()):
                templates.setdefault(path.name, []).append(item.name)
    return templates


def get_tasks() -> dict[str, list[str]]:
    """Return each bundled task mapped to the methods it provides."""
    return _list_templates(TASK_DIR)


def get_boilerplate() -> dict[str, list[str]]:
    """Return each bundled boilerplate mapped to the files it provides."""
    return _list_templates(BOILERPLATE_DIR, include_file=True)
