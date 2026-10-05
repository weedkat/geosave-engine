"""Discover bundled workspace starters and optional scaffolds."""

from pathlib import Path

_EXCLUDED_TEMPLATE_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})

TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
COMMON_DIR = TEMPLATES_DIR / "common"
WORKSPACES_DIR = TEMPLATES_DIR / "workspaces"
SCAFFOLDS_DIR = TEMPLATES_DIR / "scaffolds"


def get_workspaces() -> list[str]:
    """Return the available workspace starter names."""
    return sorted(
        path.name
        for path in WORKSPACES_DIR.iterdir()
        if path.is_dir() and path.name not in _EXCLUDED_TEMPLATE_NAMES
    )


def get_scaffolds() -> dict[str, list[str]]:
    """Return optional scaffold files grouped by destination directory."""
    return {
        path.name: sorted(
            item.name
            for item in path.iterdir()
            if item.is_file() and item.name not in _EXCLUDED_TEMPLATE_NAMES
        )
        for path in sorted(SCAFFOLDS_DIR.iterdir())
        if path.is_dir() and path.name not in _EXCLUDED_TEMPLATE_NAMES
    }
