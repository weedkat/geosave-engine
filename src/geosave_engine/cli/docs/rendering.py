from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

from rich import box
from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from geosave_engine.cli.docs.parser import (
    build_callable_signature,
    build_has_var_args,
    collect_mode_signatures,
    init_parameters_from_file,
    underlying_model_signature,
)
from geosave_engine.cli.search import InterfaceInfo, ModelInfo


_TYPE_TOKENS = {"str", "int", "float", "bool", "Optional"}


def format_annotation(annotation: Any) -> str:
    """Render a type annotation to a short, human-readable string."""
    if annotation is inspect.Parameter.empty:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__name__
    text = str(annotation).replace("typing.", "").replace("collections.abc.", "")
    return text


def extract_annotation_metadata(annotation: Any) -> str:
    """Pull metadata out of `Annotated[...]` wrappers, if any."""
    if annotation is inspect.Parameter.empty:
        return ""

    metadata: list[str] = []
    args = getattr(annotation, "__metadata__", None)
    if args:
        metadata.extend(str(item) for item in args)
        return "; ".join(metadata)

    annotation_text = str(annotation)
    if "Annotated[" in annotation_text:
        inner = annotation_text.split("Annotated[", maxsplit=1)[1].rstrip("]")
        parts = [part.strip() for part in inner.split(",")]
        if len(parts) > 1:
            metadata.extend(parts[1:])

    return "; ".join(metadata)


def highlight_type(annotation: str) -> Text:
    """Colourise a type annotation string for Rich output."""
    text = Text()
    for part in re.split(r"([A-Za-z_][A-Za-z0-9_]*)", annotation):
        if not part:
            continue
        if part in _TYPE_TOKENS:
            text.append(part, style="green")
        elif re.match(r"[A-Za-z_][A-Za-z0-9_]*", part):
            text.append(part, style="bright_cyan")
        else:
            text.append(part, style="white")
    return text


def build_section(*rows: RenderableType) -> Table:
    """Wrap a vertical stack of renderables in a single-column Rich table."""
    table = Table(
        show_header=False,
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
        expand=False,
        show_lines=True,
    )
    table.add_column("value")
    for row in rows:
        if isinstance(row, (Text, Table)):
            table.add_row(row)
        else:
            table.add_row(Text(str(row), style="white"))
    return table


def build_arguments_table(
    signature: inspect.Signature | None,
    *,
    error_message: str | None = None,
    exclude_names: set[str] | None = None,
) -> Table:
    """Render a Rich table of `name / type / default / metadata` rows."""
    table = Table(title="", header_style="bold bright_white")
    table.add_column("Parameter", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Default / Value", style="magenta")
    table.add_column("Metadata", style="cyan")

    exclude = exclude_names or set()

    if signature is None:
        msg = (
            f"unavailable ({error_message})"
            if error_message
            else "unavailable (signature lookup failed)"
        )
        table.add_row("args", Text("-", style="white"), Text("-", style="white"), Text(msg, style="yellow"))
        return table

    for param in signature.parameters.values():
        if param.name in exclude or param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        type_text = highlight_type(format_annotation(param.annotation))
        default_text: Text | str
        if param.default is inspect.Parameter.empty:
            default_text = Text("required", style="bright_yellow")
        else:
            default_text = repr(param.default)

        metadata = extract_annotation_metadata(param.annotation)
        table.add_row(Text(param.name, style="yellow"), type_text, default_text, metadata or "-")

    return table


def build_init_arguments_table(file_path: Path, class_name: str) -> Table:
    """Render the `__init__` argument table for classes inside a template file."""
    table = Table(title="", header_style="bold bright_white")
    table.add_column("Parameter", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Metadata", style="cyan")
    table.add_column("Default / Value", style="magenta")

    rows = init_parameters_from_file(file_path, class_name)
    if not rows:
        table.add_row(
            "args",
            Text("-", style="white"),
            Text("-", style="white"),
            Text("unavailable (__init__ not found or parse failed)", style="yellow"),
        )
        return table

    for row in rows:
        default_value = row["default"]
        default_render: Text | str = (
            Text("required", style="bright_yellow") if default_value is None else default_value
        )
        table.add_row(
            Text(str(row["name"]), style="yellow"),
            highlight_type(str(row["annotation"])),
            row["metadata"] or "-",
            default_render,
        )

    return table


def build_model_meta_section(info: ModelInfo) -> Table:
    """Render the full model doc panel (header, description, tasks, args).

    Shows two parameter tables when the model's `build` classmethod accepts
    `*args/**kwargs`: one for the explicit `build` parameters, and one for
    the underlying model's constructor parameters (the ones forwarded through
    `*args/**kwargs`).
    """
    model_name = Text(info.class_name, style="bold cyan")
    class_path_block = Text(f"class_path: {info.class_path}", style="cyan")

    description_block = Text()
    description_block.append(info.docstring or "No class docstring available.", style="white")
    description_block.append("\n")
    description_block.append(info.doc_link or "https://docs.com", style="blue underline")

    if info.task_map:
        tasks_text = "\n".join(
            f"{task} / {', '.join(methods) if methods else 'all'}"
            for task, methods in info.task_map.items()
        )
        tasks_block: RenderableType = Text(tasks_text, style="green")
    else:
        tasks_block = Text("No task/method metadata available", style="yellow")

    # Build parameters (always from the `build` classmethod)
    build_error: str | None = None
    build_sig = None
    try:
        build_sig = build_callable_signature(info.module_path, info.class_name)
        if build_sig is None:
            build_error = "no build classmethod found"
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError, OSError) as exc:
        build_error = f"{type(exc).__name__}: {exc}"

    # Decide whether build has any explicit (non-var) params
    build_has_explicit = build_sig is not None and any(
        param.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for param in build_sig.parameters.values()
    )
    build_forwards_all = build_sig is not None and build_has_var_args(build_sig) and not build_has_explicit

    rows: list[RenderableType] = [model_name, class_path_block, description_block, tasks_block]

    if build_forwards_all:
        # BaseModel.build style: only *args/**kwargs — skip the empty table
        rows.append(Text("build parameters", style="bold yellow"))
        rows.append(
            Text(
                "All parameters are forwarded to the model (see model parameters below).",
                style="dim white",
            )
        )
    else:
        rows.append(Text("build parameters", style="bold yellow"))
        rows.append(build_arguments_table(build_sig, error_message=build_error))

    # If build forwards *args/**kwargs, also show the underlying model's params
    if build_sig is not None and build_has_var_args(build_sig):
        model_sig_error: str | None = None
        model_sig = None
        try:
            model_sig = underlying_model_signature(info.module_path, info.class_name)
            if model_sig is None:
                model_sig_error = "no model/loss/optimizer attribute found"
        except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError, OSError) as exc:
            model_sig_error = f"{type(exc).__name__}: {exc}"

        rows.append(Text("model parameters (forwarded via *args / **kwargs)", style="bold yellow"))
        rows.append(build_arguments_table(model_sig, error_message=model_sig_error))

    return build_section(*rows)


def build_interface_meta_section(info: InterfaceInfo, kind: str) -> Table:
    """Render a loss/optimizer doc panel without importing torch.

    Uses AST-extracted metadata: class name, docstring, doc link, the
    target class reference (e.g. `torch.optim.AdamW`), and mode signatures.
    """
    class_name_text = Text(info.class_name, style="bold cyan")
    class_path_text = Text(f"class_path: {info.class_path}", style="cyan")

    description_block = Text()
    description_block.append(info.docstring or f"No class docstring available.", style="white")
    if info.doc_link:
        description_block.append("\n")
        description_block.append(info.doc_link, style="blue underline")

    target_label = "loss" if kind == "loss" else "optimizer"
    target_block = Text()
    target_block.append(f"{target_label}: ", style="bold yellow")
    target_block.append(info.target_ref or "unknown", style="bright_cyan")

    rows: list[RenderableType] = [class_name_text, class_path_text, description_block, target_block]

    for mode_name, mode_doc, param_rows in collect_mode_signatures(info.file_path, info.class_name):
        rows.append(Text(f'mode="{mode_name}"', style="bold green"))
        if mode_doc:
            rows.append(Text(mode_doc.splitlines()[0], style="white"))
        rows.append(_build_mode_params_table(param_rows))

    return build_section(*rows)


def _build_mode_params_table(param_rows: list[dict]) -> Table:
    table = Table(title="", header_style="bold bright_white")
    table.add_column("Parameter", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Default / Value", style="magenta")
    table.add_column("Metadata", style="cyan")

    for row in param_rows:
        default_val = row["default"]
        default_text: Text | str = (
            Text("required", style="bright_yellow") if default_val is None else str(default_val)
        )
        table.add_row(
            Text(str(row["name"]), style="yellow"),
            highlight_type(str(row["annotation"])),
            default_text,
            row.get("metadata") or "-",
        )
    return table
