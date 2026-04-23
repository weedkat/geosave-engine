from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

from geosave_engine.cli.search import InterfaceInfo, ModelInfo


def module_path_for_file(file_path: Path, src_root: Path) -> str:
    """Convert a `.py` file path into its dotted module path relative to `src_root`."""
    relative = file_path.relative_to(src_root).with_suffix("")
    return ".".join(relative.parts)


def extract_doc_link(value: Any) -> str | None:
    """Return the first string from `doc_link`/`doc_links` attributes."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return value[0]
    return None


def collect_model_infos(models_path: Path, src_root: Path) -> dict[str, ModelInfo]:
    """AST-parse every `build.py` under `models_path` into `ModelInfo` objects."""
    infos: dict[str, ModelInfo] = {}

    for build_file in models_path.glob("**/build.py"):
        module_path = module_path_for_file(build_file, src_root)
        class_module_path = (
            module_path[: -len(".build")] if module_path.endswith(".build") else module_path
        )

        with open(build_file, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=str(build_file))

        for class_node in tree.body:
            if not isinstance(class_node, ast.ClassDef):
                continue

            task_map, doc_link = _extract_class_metadata(class_node)
            infos[class_node.name] = ModelInfo(
                class_name=class_node.name,
                class_path=f"{class_module_path}.{class_node.name}",
                task_map=task_map,
                docstring=inspect.cleandoc(ast.get_docstring(class_node) or ""),
                doc_link=doc_link,
                module_path=module_path,
            )

    return infos


def _extract_class_metadata(
    class_node: ast.ClassDef,
) -> tuple[dict[str, list[str]], str | None]:
    task_map: dict[str, list[str]] = {}
    doc_link: str | None = None

    for stmt in class_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue

        target = stmt.targets[0].id
        if target == "tasks" and isinstance(stmt.value, ast.Dict):
            task_map = _parse_tasks_dict(stmt.value)
        elif target in {"doc_link", "doc_links"}:
            try:
                doc_link = extract_doc_link(ast.literal_eval(stmt.value))
            except Exception:
                if isinstance(stmt.value, ast.List):
                    for elem in stmt.value.elts:
                        if isinstance(elem, ast.Constant) and isinstance(elem.value, str):
                            doc_link = elem.value
                            break

    return task_map, doc_link


def _parse_tasks_dict(node: ast.Dict) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for key, value in zip(node.keys, node.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        methods: list[str] = []
        if isinstance(value, ast.List):
            for elem in value.elts:
                if isinstance(elem, ast.Constant) and isinstance(elem.value, str):
                    methods.append(elem.value)
        parsed[key.value] = methods
    return parsed


def collect_interface_infos(
    kind: str,
    root: Path,
    src_root: Path,
) -> dict[str, InterfaceInfo]:
    """AST-parse every `.py` under `root` and return `{class_name: InterfaceInfo}` entries
    that define the expected `loss`/`optimizer` attribute.

    No modules are imported — torch is never touched.
    """
    attr_name = "loss" if kind == "loss" else "optimizer"
    infos: dict[str, InterfaceInfo] = {}

    for file_path in root.glob("**/*.py"):
        if file_path.name == "__init__.py":
            continue

        module_path = module_path_for_file(file_path, src_root)

        with open(file_path, "r", encoding="utf-8") as handle:
            try:
                tree = ast.parse(handle.read(), filename=str(file_path))
            except SyntaxError:
                continue

        for class_node in tree.body:
            if not isinstance(class_node, ast.ClassDef):
                continue

            target_ref: str | None = None
            doc_link: str | None = None

            for stmt in class_node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    continue
                name = stmt.targets[0].id
                if name == attr_name:
                    target_ref = _ast_to_text(stmt.value)
                elif name in {"doc_link", "doc_links"}:
                    try:
                        doc_link = extract_doc_link(ast.literal_eval(stmt.value))
                    except Exception:
                        if isinstance(stmt.value, ast.List):
                            for elem in stmt.value.elts:
                                if isinstance(elem, ast.Constant) and isinstance(elem.value, str):
                                    doc_link = elem.value
                                    break

            if target_ref is None:
                continue

            infos[class_node.name] = InterfaceInfo(
                class_name=class_node.name,
                class_path=f"{module_path}.{class_node.name}",
                module_path=module_path,
                docstring=inspect.cleandoc(ast.get_docstring(class_node) or ""),
                doc_link=doc_link,
                target_ref=target_ref,
                file_path=file_path,
            )

    return infos


def collect_mode_signatures(
    file_path: Path,
    class_name: str,
) -> list[tuple[str, str | None, list[dict[str, Any]]]]:
    """Return `(method_name, docstring, [param_rows])` for public classmethods.

    AST-only — no import, no torch.  Skips `build`, `_private`, and `__dunder__` names.
    """
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as handle:
        try:
            tree = ast.parse(handle.read(), filename=str(file_path))
        except SyntaxError:
            return []

    class_node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name),
        None,
    )
    if class_node is None:
        return []

    modes: list[tuple[str, str | None, list[dict[str, Any]]]] = []

    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_") or node.name == "build":
            continue
        if not any(
            isinstance(d, ast.Name) and d.id == "classmethod"
            or isinstance(d, ast.Attribute) and d.attr == "classmethod"
            for d in node.decorator_list
        ):
            continue

        docstring = inspect.cleandoc(ast.get_docstring(node) or "")
        params = _extract_function_params(node, skip_first=True)
        modes.append((node.name, docstring or None, params))

    return modes


def _extract_function_params(
    func_node: ast.FunctionDef,
    skip_first: bool = False,
) -> list[dict[str, Any]]:
    """Extract parameter rows from an AST function node."""
    rows: list[dict[str, Any]] = []
    args = func_node.args
    default_offset = len(args.args) - len(args.defaults)

    for idx, arg in enumerate(args.args):
        if skip_first and idx == 0:
            continue
        default_value: Any = None
        if idx >= default_offset:
            default_value = _ast_to_text(args.defaults[idx - default_offset]) or "..."
        rows.append({
            "name": arg.arg,
            "annotation": _ast_to_text(arg.annotation) or "Any",
            "metadata": _ast_metadata_text(arg.annotation),
            "default": default_value,
        })

    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        rows.append({
            "name": kw_arg.arg,
            "annotation": _ast_to_text(kw_arg.annotation) or "Any",
            "metadata": _ast_metadata_text(kw_arg.annotation),
            "default": _ast_to_text(kw_default) if kw_default is not None else None,
        })

    if args.vararg:
        rows.append({"name": f"*{args.vararg.arg}", "annotation": "Any", "metadata": "", "default": "..."})
    if args.kwarg:
        rows.append({"name": f"**{args.kwarg.arg}", "annotation": "Any", "metadata": "", "default": "..."})

    return rows


def build_callable_signature(module_path: str, class_name: str) -> inspect.Signature | None:
    """Return the `build` classmethod signature (always, even when it has *args/**kwargs).

    Returns None if the class has no `build` method.
    """
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    build_callable = getattr(cls, "build", None)
    if not callable(build_callable):
        return None
    return inspect.signature(build_callable)


def build_has_var_args(sig: inspect.Signature) -> bool:
    """Return True if the signature contains *args or **kwargs."""
    return any(
        param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for param in sig.parameters.values()
    )


def underlying_model_signature(module_path: str, class_name: str) -> inspect.Signature | None:
    """Return the signature of the underlying model/loss/optimizer class.

    Used to expand *args/**kwargs from a `build` method into their actual parameters.
    """
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    for attr in ("model", "loss", "optimizer"):
        target = getattr(cls, attr, None)
        if target is not None:
            return inspect.signature(target)
    return None


def _ast_to_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _ast_metadata_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Subscript):
        base = _ast_to_text(node.value)
        if base and base.endswith("Annotated"):
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                parts = [
                    text
                    for text in (_ast_to_text(element) for element in slice_node.elts[1:])
                    if text
                ]
                return "; ".join(parts)
    return ""


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


def init_parameters_from_file(file_path: Path, class_name: str) -> list[dict[str, Any]]:
    """Return `__init__` parameter rows for the template's docs view."""
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

    rows: list[dict[str, Any]] = []
    args = init_node.args
    default_offset = len(args.args) - len(args.defaults)

    for idx, arg in enumerate(args.args):
        if arg.arg == "self":
            continue
        default_value: Any
        if idx >= default_offset:
            default_value = _ast_to_text(args.defaults[idx - default_offset]) or "..."
        else:
            default_value = None
        rows.append(
            {
                "name": arg.arg,
                "annotation": _ast_to_text(arg.annotation) or "Any",
                "metadata": _ast_metadata_text(arg.annotation),
                "default": default_value,
            }
        )

    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        rows.append(
            {
                "name": kw_arg.arg,
                "annotation": _ast_to_text(kw_arg.annotation) or "Any",
                "metadata": _ast_metadata_text(kw_arg.annotation),
                "default": _ast_to_text(kw_default) if kw_default is not None else None,
            }
        )

    return rows
