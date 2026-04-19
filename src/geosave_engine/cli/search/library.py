from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata extracted from a model builder's source file."""

    class_name: str
    class_path: str
    task_map: dict[str, list[str]]
    docstring: str
    doc_link: str | None
    module_path: str


@dataclass(frozen=True)
class InterfaceInfo:
    """Static metadata extracted from a loss/optimizer wrapper source file."""

    class_name: str
    class_path: str
    module_path: str
    docstring: str
    doc_link: str | None
    target_ref: str | None  # AST-unparsed value, e.g. "torch.optim.AdamW"
    file_path: Path


def discover_tasks(templates_path: Path) -> dict[str, list[str]]:
    """Walk `templates_path` and return `{task_name: [method, ...]}`.

    `common/` and dunder/pycache folders are excluded. Folder names are
    normalised: underscores become spaces, lower-cased.
    """
    tasks: dict[str, list[str]] = {}
    if not templates_path.exists():
        return tasks

    for folder in templates_path.iterdir():
        if not folder.is_dir() or folder.name in {"__pycache__", "common"}:
            continue

        methods = [
            method.name.replace("_", " ").lower()
            for method in folder.iterdir()
            if method.is_dir() and method.name != "__pycache__"
        ]
        tasks[folder.name.replace("_", " ").lower()] = methods

    return tasks


def discover_model_names(task: str, method: str, models_package_path: Path) -> list[str]:
    """Return model class names whose `tasks` dict declares `task` (and `method`, if any).

    Parses each `build.py` with `ast` to avoid importing heavy model modules.
    """
    class_names: list[str] = []

    for build_file in models_package_path.glob("**/build.py"):
        with open(build_file, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if _class_matches(node, task, method):
                class_names.append(node.name)

    return class_names


def _class_matches(class_node: ast.ClassDef, task: str, method: str) -> bool:
    for stmt in class_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if not (isinstance(target, ast.Name) and target.id == "tasks"):
                continue
            if not isinstance(stmt.value, ast.Dict):
                continue

            for key, value in zip(stmt.value.keys, stmt.value.values):
                if not (isinstance(key, ast.Constant) and key.value == task):
                    continue
                if not isinstance(value, ast.List):
                    continue
                methods = [
                    elt.value
                    for elt in value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                if not method or not methods or method in methods:
                    return True
    return False
