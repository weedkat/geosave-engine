from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import questionary
import typer
from rich import box
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from geosave_engine.cli.metadata import (
    losses_root,
    models_root,
    normalize_slug,
    optimizers_root,
    resolve_from_choices,
    templates_root,
)
from geosave_engine.cli.services.docs.docs_parsing_service import DocsParsingService
from geosave_engine.cli.services.search.project_search import ProjectSearch


console = Console()


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__name__

    text = str(annotation)
    text = text.replace("typing.", "")
    text = text.replace("collections.abc.", "")
    return text


def _extract_metadata_from_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""

    metadata: list[str] = []
    args = getattr(annotation, "__metadata__", None)
    if args:
        metadata.extend(str(item) for item in args)

    annotation_text = str(annotation)
    if "Annotated[" in annotation_text and not metadata:
        inner = annotation_text.split("Annotated[", maxsplit=1)[1].rstrip("]")
        parts = [part.strip() for part in inner.split(",")]
        if len(parts) > 1:
            metadata.extend(parts[1:])

    return "; ".join(metadata)


def _format_default(value: Any) -> str:
    return repr(value)


def _highlight_type(annotation: str) -> Text:
    text = Text()
    type_tokens = {"str", "int", "float", "bool", "Optional"}
    parts = re.split(r"([A-Za-z_][A-Za-z0-9_]*)", annotation)
    for part in parts:
        if not part:
            continue
        if part in type_tokens:
            text.append(part, style="green")
        elif re.match(r"[A-Za-z_][A-Za-z0-9_]*", part):
            text.append(part, style="bright_cyan")
        else:
            text.append(part, style="white")
    return text


def _build_doc_cli(*rows: RenderableType) -> Table:
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


def _build_arguments_table(
    signature: inspect.Signature | None,
    error_message: str | None = None,
    exclude_names: set[str] | None = None,
) -> Table:
    table = Table(title="", header_style="bold bright_white")
    table.add_column("Parameter", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Default / Value", style="magenta")
    table.add_column("Metadata", style="cyan")
    exclude_names = exclude_names or set()

    if signature is None:
        error_text = (
            f"unavailable ({error_message})"
            if error_message
            else "unavailable (signature lookup failed)"
        )
        table.add_row(
            "args",
            Text("-", style="white"),
            Text("-", style="white"),
            Text(error_text, style="yellow"),
        )
        return table

    for param in signature.parameters.values():
        if param.name in exclude_names:
            continue
        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        name = param.name

        type_text = _highlight_type(_format_annotation(param.annotation))
        if param.default is inspect.Parameter.empty:
            default_text: Text | str = Text("required", style="bright_yellow")
        else:
            default_text = _format_default(param.default)

        metadata_text = _extract_metadata_from_annotation(param.annotation)

        table.add_row(Text(name, style="yellow"), type_text, metadata_text or "-", default_text)

    return table


def _build_init_arguments_table_from_file(file_path: Path, class_name: str) -> Table:
    table = Table(title="", header_style="bold bright_white")
    table.add_column("Parameter", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Metadata", style="cyan")
    table.add_column("Default / Value", style="magenta")

    rows = DocsParsingService.init_parameters_from_file(file_path, class_name)
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
        if default_value is None:
            default_render: Text | str = Text("required", style="bright_yellow")
        else:
            default_render = default_value
        table.add_row(
            Text(str(row["name"]), style="yellow"),
            _highlight_type(str(row["annotation"])),
            row["metadata"] or "-",
            default_render,
        )

    return table


def _format_mode_header(factory_name: str, mode_name: str) -> str:
    return f'{factory_name} (mode="{mode_name}")'


def _build_model_meta_table(info) -> Table:
    model_name = Text(info.class_name, style="bold cyan")

    class_path_block = Text(f"class_path: {info.class_path}", style="cyan")

    description_block = Text()
    description_block.append(info.docstring or "No class docstring available.", style="white")
    description_block.append("\n")
    description_block.append(info.doc_link or "https://docs.com", style="blue underline")

    if info.task_map:
        task_lines: list[str] = []
        for task_name, methods in info.task_map.items():
            method_text = ", ".join(methods) if methods else "all"
            task_lines.append(f"{task_name} / {method_text}")
        tasks_block: RenderableType = Text("\n".join(task_lines), style="green")
    else:
        tasks_block = Text("No task/method metadata available", style="yellow")

    arg_error: str | None = None
    try:
        signature = DocsParsingService.callable_signature(info.module_path, info.class_name)
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError, OSError) as exc:
        signature = None
        arg_error = f"{type(exc).__name__}: {exc}"

    parameters_table = _build_arguments_table(signature, arg_error)
    return _build_doc_cli(
        model_name,
        class_path_block,
        description_block,
        tasks_block,
        parameters_table,
    )


def _show_lightning_module_docs() -> None:
    template_file = templates_root() / "semantic_segmentation" / "supervised" / "src" / "lightning_module.py"
    class_name = "GeosaveLightningModule"
    doc = DocsParsingService.class_docstring_from_file(template_file, class_name)
    parameters_table = _build_init_arguments_table_from_file(template_file, class_name)

    global_keys_block = Text(
        "# Global settings\nmodel:\n  model_config:\n  optimizer_config:\n  loss_config:",
        style="green",
    )
    local_keys_block = Text(
        "# Local settings\nfit:\n  model:\n    model_config:\n    optimizer_config:\n    loss_config:",
        style="green",
    )
    desc_block = Text(doc.splitlines()[0] if doc else "No class docstring available.", style="white")

    console.print(
        _build_doc_cli(
            Text("LightningModule", style="bold cyan"),
            desc_block,
            global_keys_block,
            local_keys_block,
            Text("Parameters are declared in your custom lightning module implementation.", style="white"),
            Text("constructor parameters", style="bold yellow"),
            parameters_table,
        )
    )


def _show_data_module_docs() -> None:
    template_file = templates_root() / "semantic_segmentation" / "supervised" / "src" / "data_module.py"
    class_name = "GeosaveDataModule"
    doc = DocsParsingService.class_docstring_from_file(template_file, class_name)
    parameters_table = _build_init_arguments_table_from_file(template_file, class_name)

    global_keys_block = Text("# Global settings\ndata:\n  data_dir:", style="green")
    local_keys_block = Text("# Local settings\ndata:\n  fit:\n    data_dir:", style="green")
    desc_block = Text(doc.splitlines()[0] if doc else "No class docstring available.", style="white")

    console.print(
        _build_doc_cli(
            Text("LightningDataModule", style="bold cyan"),
            desc_block,
            global_keys_block,
            local_keys_block,
            Text(
                "Parameters are declared in your custom data module implementation (for example data_modules.py).",
                style="white",
            ),
            Text("constructor parameters", style="bold yellow"),
            parameters_table,
        )
    )


def _show_trainer_docs() -> None:
    try:
        from lightning import Trainer
    except (ImportError, ModuleNotFoundError):
        try:
            from pytorch_lightning import Trainer
        except (ImportError, ModuleNotFoundError):
            console.print(
                _build_doc_cli(
                    Text("Trainer", style="bold cyan"),
                    Text("Unable to import lightning Trainer in current environment.", style="yellow"),
                )
            )
            return

    trainer_doc = inspect.getdoc(Trainer) or "PyTorch Lightning Trainer configuration."
    trainer_doc_short = trainer_doc.split("\n\n", maxsplit=1)[0]
    trainer_sig_table = _build_arguments_table(inspect.signature(Trainer))

    global_keys_block = Text("# Global settings\ntrainer:\n  max_epochs:\n  callbacks:\n  logger:", style="green")
    local_keys_block = Text("# Local settings\nfit:\n  trainer:\ntest:\n  trainer:\npredict:\n  trainer:", style="green")

    console.print(
        _build_doc_cli(
            Text("Trainer", style="bold cyan"),
            Text(trainer_doc_short, style="white"),
            global_keys_block,
            local_keys_block,
            Text("trainer parameters", style="bold yellow"),
            trainer_sig_table,
        )
    )


def _show_registry_docs(kind: str, selected_name: str | None = None) -> None:
    title = "Loss" if kind == "loss" else "Optimizer"
    root = losses_root() if kind == "loss" else optimizers_root()
    src_root = Path(__file__).resolve().parents[4]
    registry = DocsParsingService.collect_interface_registry(kind, root, src_root)

    if not registry:
        console.print(
            _build_doc_cli(
                Text(title, style="bold cyan"),
                Text("No interface classes discovered in current environment.", style="yellow"),
            )
        )
        return

    names = sorted(registry.keys())
    if not selected_name:
        selected_name = questionary.select(
            f"Choose {title.lower()}:",
            choices=[questionary.Choice(name, value=name) for name in names],
        ).ask()
        if not selected_name:
            raise typer.Exit(1)

    resolved = resolve_from_choices(selected_name, names)
    if not resolved:
        typer.secho(f"Unknown {kind}: {selected_name}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    cls = registry[resolved]
    class_doc = inspect.getdoc(cls) or "No class docstring available."
    class_doc_short = class_doc.split("\n\n", maxsplit=1)[0]

    target_attr = "loss" if kind == "loss" else "optimizer"
    target_callable = getattr(cls, target_attr)
    target_sig_table = _build_arguments_table(
        inspect.signature(target_callable),
        exclude_names={"model"} if kind == "optimizer" else None,
    )

    mode_rows: list[RenderableType] = []
    for mode_name, mode_callable in DocsParsingService.collect_mode_callables(cls):
        mode_sig_table = _build_arguments_table(
            inspect.signature(mode_callable),
            exclude_names={"model"} if kind == "optimizer" else None,
        )
        mode_doc = inspect.getdoc(mode_callable) or "No mode docstring available."
        mode_doc_short = mode_doc.split("\n\n", maxsplit=1)[0]
        mode_header = _format_mode_header(cls.__name__, mode_name)
        mode_rows.extend([Text(mode_header, style="bold green"), Text(mode_doc_short, style="white"), mode_sig_table])

    doc_links = getattr(cls, "doc_links", None)
    doc_link = DocsParsingService.extract_doc_link(doc_links)
    doc_link_text = Text(doc_link or "https://docs.com", style="blue underline")

    console.print(
        _build_doc_cli(
            Text(cls.__name__, style="bold cyan"),
            Text(class_doc_short, style="white"),
            doc_link_text,
            Text(f"base {target_attr} signature", style="bold yellow"),
            target_sig_table,
            *mode_rows,
        )
    )


def _show_template_docs(task_arg: str | None = None) -> None:
    task_map = ProjectSearch.get_tasks(templates_root())
    if not task_map:
        console.print(
            _build_doc_cli(
                Text("Templates", style="bold cyan"),
                Text("No templates discovered.", style="yellow"),
            )
        )
        return

    tasks = sorted(task_map.keys())
    selected_task: str | None = None
    if task_arg:
        selected_task = resolve_from_choices(task_arg, tasks)
        if not selected_task:
            typer.secho(f"Unknown task: {task_arg}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

    table = Table(title="", header_style="bold bright_white")
    table.add_column("Task", style="yellow")
    table.add_column("Methods", style="green")

    for task in tasks:
        if selected_task and task != selected_task:
            continue
        methods = task_map[task]
        table.add_row(task, ", ".join(methods) if methods else "-")

    console.print(
        _build_doc_cli(
            Text("Templates", style="bold cyan"),
            Text("Available project templates by task and method.", style="white"),
            table,
        )
    )


def _show_model_docs(task_arg: str | None, method_arg: str | None, model_arg: str | None) -> None:
    task_map = ProjectSearch.get_tasks(templates_root())
    src_root = Path(__file__).resolve().parents[4]
    info_map = DocsParsingService.collect_model_infos(models_root(), src_root)

    tasks = sorted(task_map.keys())
    if task_arg:
        task = resolve_from_choices(task_arg, tasks)
        if not task:
            typer.secho(f"Unknown task: {task_arg}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    else:
        task = questionary.select(
            "Choose task:",
            choices=[questionary.Choice(task_name, value=task_name) for task_name in tasks],
        ).ask()
        if not task:
            raise typer.Exit(1)

    methods = task_map[task]
    method = ""
    if methods:
        if method_arg:
            resolved = resolve_from_choices(method_arg, methods)
            if not resolved:
                typer.secho(
                    f"Unknown method '{method_arg}' for task '{task}'",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(1)
            method = resolved
        else:
            method = questionary.select(
                "Choose method:",
                choices=[questionary.Choice(method_name, value=method_name) for method_name in methods],
            ).ask()
            if not method:
                raise typer.Exit(1)

    discovered_models = ProjectSearch.get_model_list(task, method, models_root())
    model_names = sorted(name for name in discovered_models if name in info_map)
    if not model_names:
        typer.secho(
            f"No models available for task='{task}' method='{method or 'all'}'",
            fg=typer.colors.YELLOW,
        )
        return

    if model_arg:
        model_name = resolve_from_choices(model_arg, model_names)
        if not model_name:
            typer.secho(f"Unknown model: {model_arg}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    else:
        model_name = questionary.select(
            "Choose model:",
            choices=[questionary.Choice(name, value=name) for name in model_names],
        ).ask()
        if not model_name:
            raise typer.Exit(1)

    info = info_map.get(model_name)
    if info:
        console.print(_build_model_meta_table(info))
    else:
        typer.echo("Model metadata unavailable.")


def show_docs(
    section: str | None = None,
    arg1: str | None = None,
    arg2: str | None = None,
    arg3: str | None = None,
) -> None:
    if not section:
        section = questionary.select(
            "Choose Geosave docs:",
            choices=[
                questionary.Choice("LightningModule", value="lightningmodule"),
                questionary.Choice("LightningDataModule", value="datamodule"),
                questionary.Choice("Trainer", value="trainer"),
                questionary.Choice("Templates", value="templates"),
                questionary.Choice("Models", value="model"),
                questionary.Choice("Losses", value="loss"),
                questionary.Choice("Optimizers", value="optimizer"),
            ],
        ).ask()
        if not section:
            raise typer.Exit(1)

    section_norm = normalize_slug(section)
    if section_norm in {"lightningmodule", "lightning module"}:
        _show_lightning_module_docs()
        return
    if section_norm in {"datamodule", "data module"}:
        _show_data_module_docs()
        return
    if section_norm in {"trainer"}:
        _show_trainer_docs()
        return
    if section_norm in {"template", "templates"}:
        _show_template_docs(arg1)
        return
    if section_norm in {"model", "models"}:
        _show_model_docs(arg1, arg2, arg3)
        return
    if section_norm in {"loss", "losses"}:
        _show_registry_docs("loss", selected_name=arg1)
        return
    if section_norm in {"optimizer", "optimizers"}:
        _show_registry_docs("optimizer", selected_name=arg1)
        return

    typer.secho(f"Unknown docs section: {section}", fg=typer.colors.RED, err=True)
    typer.secho(
        "Valid sections: lightningmodule, datamodule, trainer, templates, model, loss, optimizer",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(1)
