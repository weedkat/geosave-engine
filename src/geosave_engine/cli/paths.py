from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Absolute path to the installed `geosave_engine` package."""
    return Path(__file__).resolve().parents[1]


def templates_root() -> Path:
    """Root of the shipped workspace templates (siblings of the package)."""
    return package_root().parent / "templates" / "workspace"


def plugins_root() -> Path:
    """Root of the shipped plugins (siblings of the package)."""
    return package_root().parent / "templates" / "plugins"


def models_root() -> Path:
    return package_root() / "ml" / "models"


def losses_root() -> Path:
    return package_root() / "ml" / "losses"


def optimizers_root() -> Path:
    return package_root() / "ml" / "optimizers"


def src_root() -> Path:
    """Parent of the package — used when translating file paths to module paths."""
    return package_root().parent
