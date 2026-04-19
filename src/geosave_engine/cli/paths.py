from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Absolute path to the installed `geosave_engine` package."""
    return Path(__file__).resolve().parents[1]


def templates_root() -> Path:
    """Root of the shipped workspace templates (siblings of the package)."""
    return package_root().parent / "templates"


def models_root() -> Path:
    return package_root() / "models"


def losses_root() -> Path:
    return package_root() / "losses"


def optimizers_root() -> Path:
    return package_root() / "optimizers"


def src_root() -> Path:
    """Parent of the package — used when translating file paths to module paths."""
    return package_root().parent
