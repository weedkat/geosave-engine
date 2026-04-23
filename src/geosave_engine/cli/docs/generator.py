from __future__ import annotations

import inspect
from typing import Any

from rich.console import Console as RichConsole
from rich.table import Table
from rich.text import Text

from geosave_engine.cli.docs.parser import (
    class_docstring_from_file,
    collect_interface_infos,
    collect_model_infos,
)
from geosave_engine.cli.docs.rendering import (
    build_arguments_table,
    build_init_arguments_table,
    build_interface_meta_section,
    build_model_meta_section,
    build_section,
)
from geosave_engine.cli.errors import AbortedByUser, DocsError
from geosave_engine.cli.io import Prompter
from geosave_engine.cli.paths import (
    losses_root,
    models_root,
    optimizers_root,
    src_root,
    templates_root,
)
from geosave_engine.cli.search import discover_model_names, discover_tasks
from geosave_engine.utils.cli.strings import normalize_slug, resolve_from_choices


_rich_console = RichConsole()


_SECTION_CHOICES: list[tuple[str, str]] = [
    ("LightningModule", "lightningmodule"),
    ("LightningDataModule", "datamodule"),
    ("Trainer", "trainer"),
    ("Templates", "templates"),
    ("Models", "model"),
    ("Losses", "loss"),
    ("Optimizers", "optimizer"),
]


def show_docs(
    section: str | None,
    arg1: str | None,
    arg2: str | None,
    arg3: str | None,
    *,
    prompter: Prompter,
) -> None:
    """Entry point invoked by the `geosave docs` CLI command."""
    if not section:
        selected = prompter.select_mapping("Choose Geosave docs:", _SECTION_CHOICES)
        if not selected:
            raise AbortedByUser("Docs selection cancelled.")
        section = selected

    section_norm = normalize_slug(section)
    handler = _SECTION_HANDLERS.get(section_norm)
    if handler is None:
        raise DocsError(
            f"Unknown docs section: {section}. "
            "Valid sections: lightningmodule, datamodule, trainer, templates, model, loss, optimizer."
        )
    handler(arg1, arg2, arg3, prompter)


def _show_lightning_module(arg1, arg2, arg3, prompter: Prompter) -> None:
    template_file = templates_root() / "semantic_segmentation" / "supervised" / "src" / "lightning_module.py"
    class_name = "GeosaveLightningModule"
    doc = class_docstring_from_file(template_file, class_name)
    parameters_table = build_init_arguments_table(template_file, class_name)

    _rich_console.print(
        build_section(
            Text("LightningModule", style="bold cyan"),
            Text(doc.splitlines()[0] if doc else "No class docstring available.", style="white"),
            Text(
                "# Global settings\nmodel:\n  model_config:\n  optimizer_config:\n  loss_config:",
                style="green",
            ),
            Text(
                "# Local settings\nfit:\n  model:\n    model_config:\n    optimizer_config:\n    loss_config:",
                style="green",
            ),
            Text(
                "Parameters are declared in your custom lightning module implementation.",
                style="white",
            ),
            Text("constructor parameters", style="bold yellow"),
            parameters_table,
        )
    )


def _show_data_module(arg1, arg2, arg3, prompter: Prompter) -> None:
    template_file = templates_root() / "semantic_segmentation" / "supervised" / "src" / "data_module.py"
    class_name = "GeosaveDataModule"
    doc = class_docstring_from_file(template_file, class_name)
    parameters_table = build_init_arguments_table(template_file, class_name)

    _rich_console.print(
        build_section(
            Text("LightningDataModule", style="bold cyan"),
            Text(doc.splitlines()[0] if doc else "No class docstring available.", style="white"),
            Text("# Global settings\ndata:\n  data_dir:", style="green"),
            Text("# Local settings\ndata:\n  fit:\n    data_dir:", style="green"),
            Text(
                "Parameters are declared in your custom data module implementation (e.g. data_module.py).",
                style="white",
            ),
            Text("constructor parameters", style="bold yellow"),
            parameters_table,
        )
    )


def _show_trainer(arg1, arg2, arg3, prompter: Prompter) -> None:
    trainer_cls = _import_trainer()
    if trainer_cls is None:
        _rich_console.print(
            build_section(
                Text("Trainer", style="bold cyan"),
                Text("Unable to import lightning Trainer in current environment.", style="yellow"),
            )
        )
        return

    trainer_doc = inspect.getdoc(trainer_cls) or "PyTorch Lightning Trainer configuration."
    trainer_doc_short = trainer_doc.split("\n\n", maxsplit=1)[0]
    _rich_console.print(
        build_section(
            Text("Trainer", style="bold cyan"),
            Text(trainer_doc_short, style="white"),
            Text(
                "# Global settings\ntrainer:\n  max_epochs:\n  callbacks:\n  logger:",
                style="green",
            ),
            Text(
                "# Local settings\nfit:\n  trainer:\ntest:\n  trainer:\npredict:\n  trainer:",
                style="green",
            ),
            Text("trainer parameters", style="bold yellow"),
            build_arguments_table(inspect.signature(trainer_cls)),
        )
    )


def _import_trainer() -> type[Any] | None:
    try:
        from lightning import Trainer  # type: ignore

        return Trainer
    except (ImportError, ModuleNotFoundError):
        try:
            from pytorch_lightning import Trainer  # type: ignore

            return Trainer
        except (ImportError, ModuleNotFoundError):
            return None


def _show_registry(kind: str, selected_name: str | None, prompter: Prompter) -> None:
    title = "Loss" if kind == "loss" else "Optimizer"
    root = losses_root() if kind == "loss" else optimizers_root()
    infos = collect_interface_infos(kind, root, src_root())

    if not infos:
        _rich_console.print(
            build_section(
                Text(title, style="bold cyan"),
                Text("No interface classes discovered.", style="yellow"),
            )
        )
        return

    names = sorted(infos.keys())
    if not selected_name:
        selected_name = prompter.select_mapping(
            f"Choose {title.lower()}:", [(name, name) for name in names]
        )
        if not selected_name:
            raise AbortedByUser("Selection cancelled.")

    resolved = resolve_from_choices(selected_name, names)
    if not resolved:
        raise DocsError(f"Unknown {kind}: {selected_name}")

    _rich_console.print(build_interface_meta_section(infos[resolved], kind))


def _show_templates(arg1, arg2, arg3, prompter: Prompter) -> None:
    task_map = discover_tasks(templates_root())
    if not task_map:
        _rich_console.print(
            build_section(
                Text("Templates", style="bold cyan"),
                Text("No templates discovered.", style="yellow"),
            )
        )
        return

    tasks = sorted(task_map.keys())
    filter_task: str | None = None
    if arg1:
        filter_task = resolve_from_choices(arg1, tasks)
        if not filter_task:
            raise DocsError(f"Unknown task: {arg1}")

    table = Table(title="", header_style="bold bright_white")
    table.add_column("Task", style="yellow")
    table.add_column("Methods", style="green")

    for task in tasks:
        if filter_task and task != filter_task:
            continue
        methods = task_map[task]
        table.add_row(task, ", ".join(methods) if methods else "-")

    _rich_console.print(
        build_section(
            Text("Templates", style="bold cyan"),
            Text("Available project templates by task and method.", style="white"),
            table,
        )
    )


def _show_models(arg1, arg2, arg3, prompter: Prompter) -> None:
    task_map = discover_tasks(templates_root())
    info_map = collect_model_infos(models_root(), src_root())

    tasks = sorted(task_map.keys())
    task = (
        resolve_from_choices(arg1, tasks) if arg1 else None
    ) or prompter.select_mapping("Choose task:", [(name, name) for name in tasks])
    if not task:
        raise AbortedByUser("Task selection cancelled.")
    if task not in task_map:
        raise DocsError(f"Unknown task: {arg1 or task}")

    methods = task_map[task]
    method = ""
    if methods:
        if arg2:
            resolved = resolve_from_choices(arg2, methods)
            if not resolved:
                raise DocsError(f"Unknown method '{arg2}' for task '{task}'")
            method = resolved
        else:
            selected = prompter.select_mapping(
                "Choose method:", [(name, name) for name in methods]
            )
            if not selected:
                raise AbortedByUser("Method selection cancelled.")
            method = selected

    discovered = discover_model_names(task, method, models_root())
    model_names = sorted(name for name in discovered if name in info_map)
    if not model_names:
        raise DocsError(
            f"No models available for task='{task}' method='{method or 'all'}'"
        )

    if arg3:
        model_name = resolve_from_choices(arg3, model_names)
        if not model_name:
            raise DocsError(f"Unknown model: {arg3}")
    else:
        model_name = prompter.select_mapping(
            "Choose model:", [(name, name) for name in model_names]
        )
        if not model_name:
            raise AbortedByUser("Model selection cancelled.")

    info = info_map.get(model_name)
    if info is None:
        raise DocsError("Model metadata unavailable.")
    _rich_console.print(build_model_meta_section(info))


def _show_loss(arg1, arg2, arg3, prompter: Prompter) -> None:
    _show_registry("loss", arg1, prompter)


def _show_optimizer(arg1, arg2, arg3, prompter: Prompter) -> None:
    _show_registry("optimizer", arg1, prompter)


_SECTION_HANDLERS = {
    "lightningmodule": _show_lightning_module,
    "lightning module": _show_lightning_module,
    "datamodule": _show_data_module,
    "data module": _show_data_module,
    "trainer": _show_trainer,
    "templates": _show_templates,
    "template": _show_templates,
    "model": _show_models,
    "models": _show_models,
    "loss": _show_loss,
    "losses": _show_loss,
    "optimizer": _show_optimizer,
    "optimizers": _show_optimizer,
}
