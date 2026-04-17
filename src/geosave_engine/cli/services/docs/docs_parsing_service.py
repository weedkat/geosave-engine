from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import typer

from geosave_engine.cli.metadata import ModelInfo


class DocsParsingService:
    @staticmethod
    def module_path_for_file(file_path: Path, src_root: Path) -> str:
        relative = file_path.relative_to(src_root).with_suffix("")
        return ".".join(relative.parts)

    @staticmethod
    def extract_doc_link(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, str):
                return first
        return None

    @staticmethod
    def collect_model_infos(models_path: Path, src_root: Path) -> dict[str, ModelInfo]:
        infos: dict[str, ModelInfo] = {}
        for build_file in models_path.glob("**/build.py"):
            module_path = DocsParsingService.module_path_for_file(build_file, src_root)
            class_module_path = (
                module_path[: -len(".build")]
                if module_path.endswith(".build")
                else module_path
            )
            with open(build_file, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=str(build_file))

            for class_node in tree.body:
                if not isinstance(class_node, ast.ClassDef):
                    continue

                task_map: dict[str, list[str]] = {}
                doc_link: str | None = None

                for stmt in class_node.body:
                    if not isinstance(stmt, ast.Assign):
                        continue
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                        continue

                    target_name = stmt.targets[0].id
                    if target_name == "tasks" and isinstance(stmt.value, ast.Dict):
                        for key, value in zip(stmt.value.keys, stmt.value.values):
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                methods: list[str] = []
                                if isinstance(value, ast.List):
                                    for method_node in value.elts:
                                        if isinstance(method_node, ast.Constant) and isinstance(method_node.value, str):
                                            methods.append(method_node.value)
                                task_map[key.value] = methods

                    if target_name == "doc_link":
                        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                            doc_link = stmt.value.value
                        else:
                            try:
                                doc_link = DocsParsingService.extract_doc_link(ast.literal_eval(stmt.value))
                            except Exception:
                                pass

                    if target_name == "doc_links":
                        try:
                            doc_link = DocsParsingService.extract_doc_link(ast.literal_eval(stmt.value))
                        except Exception:
                            if isinstance(stmt.value, ast.List):
                                for link_node in stmt.value.elts:
                                    if isinstance(link_node, ast.Constant) and isinstance(link_node.value, str):
                                        doc_link = link_node.value
                                        break

                infos[class_node.name] = ModelInfo(
                    class_name=class_node.name,
                    class_path=f"{class_module_path}.{class_node.name}",
                    task_map=task_map,
                    docstring=inspect.cleandoc(ast.get_docstring(class_node) or ""),
                    doc_link=doc_link,
                    module_path=module_path,
                )

        return infos

    @staticmethod
    def collect_interface_registry(
        kind: str,
        root: Path,
        src_root: Path,
    ) -> dict[str, type[Any]]:
        attr_name = "loss" if kind == "loss" else "optimizer"
        registry: dict[str, type[Any]] = {}

        for file_path in root.glob("**/*.py"):
            if file_path.name == "__init__.py":
                continue

            module_path = DocsParsingService.module_path_for_file(file_path, src_root)
            try:
                module = importlib.import_module(module_path)
            except (ImportError, ModuleNotFoundError, OSError) as error:
                typer.secho(
                    f"Skipping {module_path}: {error}",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue
                if getattr(cls, attr_name, None) is None:
                    continue
                registry[cls.__name__] = cls

        return registry

    @staticmethod
    def collect_mode_callables(cls: type[Any]) -> list[tuple[str, Any]]:
        classmethod_names = [
            name
            for name, value in cls.__dict__.items()
            if isinstance(value, classmethod) and name not in {"build"} and not name.startswith("_")
        ]
        mode_names = sorted(classmethod_names)

        modes: list[tuple[str, Any]] = []
        for mode_name in mode_names:
            callable_obj = getattr(cls, mode_name, None)
            if callable(callable_obj):
                modes.append((mode_name, callable_obj))
        return modes

    @staticmethod
    def callable_signature(module_path: str, class_name: str) -> inspect.Signature | None:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        build_callable = getattr(cls, "build", None)
        if callable(build_callable):
            build_sig = inspect.signature(build_callable)
            has_generic = any(
                parameter.kind
                in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
                for parameter in build_sig.parameters.values()
            )
            if not has_generic:
                return build_sig

        callable_obj = getattr(cls, "model", None)
        if callable_obj is None:
            callable_obj = getattr(cls, "loss", None)
        if callable_obj is None:
            callable_obj = getattr(cls, "optimizer", None)
        if callable_obj is None:
            raise AttributeError(
                f"Class '{class_name}' has no build/model/loss/optimizer callable"
            )

        return inspect.signature(callable_obj)

    @staticmethod
    def ast_to_text(node: ast.AST | None) -> str | None:
        if node is None:
            return None
        return ast.unparse(node)

    @staticmethod
    def ast_metadata_text(node: ast.AST | None) -> str:
        if node is None:
            return ""

        if isinstance(node, ast.Subscript):
            base_name = DocsParsingService.ast_to_text(node.value)
            if base_name and base_name.endswith("Annotated"):
                slice_node = node.slice
                if isinstance(slice_node, ast.Tuple):
                    metadata_parts = [
                        part
                        for part in (
                            DocsParsingService.ast_to_text(part_node)
                            for part_node in slice_node.elts[1:]
                        )
                        if part
                    ]
                    return "; ".join(metadata_parts)
        return ""

    @staticmethod
    def class_docstring_from_file(file_path: Path, class_name: str) -> str | None:
        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=str(file_path))

        class_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if class_node is None:
            return None
        return inspect.cleandoc(ast.get_docstring(class_node) or "")

    @staticmethod
    @staticmethod
    def init_parameters_from_file(file_path: Path, class_name: str) -> list[dict[str, Any]]:
        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=str(file_path))

        class_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if class_node is None:
            return []

        init_node = next(
            (
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            ),
            None,
        )
        if init_node is None:
            return []

        args = init_node.args
        arg_nodes = args.args
        defaults = args.defaults
        default_offset = len(arg_nodes) - len(defaults)

        rows: list[dict[str, Any]] = []

        for idx, arg in enumerate(arg_nodes):
            if arg.arg == "self":
                continue
            annotation = DocsParsingService.ast_to_text(arg.annotation) or "Any"
            metadata = DocsParsingService.ast_metadata_text(arg.annotation)
            if idx >= default_offset:
                default_node = defaults[idx - default_offset]
                default_value: Any = DocsParsingService.ast_to_text(default_node) or "..."
            else:
                default_value = None
            rows.append(
                {
                    "name": arg.arg,
                    "annotation": annotation,
                    "metadata": metadata,
                    "default": default_value,
                }
            )

        for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
            rows.append(
                {
                    "name": kw_arg.arg,
                    "annotation": DocsParsingService.ast_to_text(kw_arg.annotation)
                    or "Any",
                    "metadata": DocsParsingService.ast_metadata_text(
                        kw_arg.annotation
                    ),
                    "default": DocsParsingService.ast_to_text(kw_default)
                    if kw_default is not None
                    else None,
                }
            )

        return rows
