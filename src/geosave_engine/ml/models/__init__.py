"""Import model adapters to register their named factories."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Sequence


def _import_all_submodules(package_name: str, package_path: Sequence[str]) -> None:
    """Recursively import every module and subpackage under one package.

    A module that fails to import raises; nothing is swallowed.

    Args:
        package_name: Dotted name of the package to walk (e.g. this package's ``__name__``).
        package_path: That package's own ``__path__``.
    """
    for module_info in pkgutil.iter_modules(package_path, prefix=f"{package_name}."):
        module = importlib.import_module(module_info.name)
        if module_info.ispkg:
            _import_all_submodules(module_info.name, module.__path__)


_import_all_submodules(__name__, __path__)
