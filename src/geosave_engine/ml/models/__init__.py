"""Importing this package recursively imports every module under it, so
every @register_model decorator runs and populates
geosave_engine.ml.registry.model.MODEL_REGISTRY — no per-file import to
remember. build_model imports this package lazily before any registry
lookup — see registry.model._resolve_stage_cls.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Sequence


def _import_all_submodules(package_name: str, package_path: Sequence[str]) -> None:
    """Recursively import every module/subpackage under one package.

    Uses ``pkgutil.iter_modules`` (lists names, imports nothing itself)
    plus a plain ``importlib.import_module`` call, not
    ``pkgutil.walk_packages`` — that helper silently swallows ImportError
    for anything that fails to import, which would hide a genuinely broken
    model file instead of raising it.

    Args:
        package_name: Dotted name of the package to walk (e.g. this package's ``__name__``).
        package_path: That package's own ``__path__``.
    """
    for module_info in pkgutil.iter_modules(package_path, prefix=f"{package_name}."):
        module = importlib.import_module(module_info.name)
        if module_info.ispkg:
            _import_all_submodules(module_info.name, module.__path__)


_import_all_submodules(__name__, __path__)
