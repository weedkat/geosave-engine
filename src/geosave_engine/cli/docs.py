from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import questionary
import typer
from rich import box
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from geosave_engine.cli.utils.parse import get_tasks


console = Console()


@dataclass
class ModelInfo:
	class_name: str
	model_callable_name: str | None
	task_map: dict[str, list[str]]
	docstring: str
	doc_link: str | None
	module_path: str
	source_path: Path


def _templates_root() -> Path:
	return Path(__file__).resolve().parents[2] / "templates"


def _models_root() -> Path:
	return Path(__file__).resolve().parents[1] / "models"


def _losses_root() -> Path:
	return Path(__file__).resolve().parents[1] / "losses"


def _optimizers_root() -> Path:
	return Path(__file__).resolve().parents[1] / "optimizers"


def _normalize_slug(value: str) -> str:
	return " ".join(value.replace("_", " ").replace("-", " ").lower().split())


def _resolve_from_choices(raw_value: str, choices: list[str]) -> str | None:
	normalized = _normalize_slug(raw_value)
	norm_map = {_normalize_slug(c): c for c in choices}
	if normalized in norm_map:
		return norm_map[normalized]

	target_tokens = set(normalized.split())
	for choice in choices:
		choice_tokens = set(_normalize_slug(choice).split())
		if target_tokens and target_tokens == choice_tokens:
			return choice
	return None


def _module_path_for_file(file_path: Path) -> str:
	src_root = Path(__file__).resolve().parents[2]
	rel = file_path.relative_to(src_root).with_suffix("")
	return ".".join(rel.parts)


def _expr_to_name(node: ast.AST) -> str | None:
	if isinstance(node, ast.Name):
		return node.id
	if isinstance(node, ast.Attribute):
		parts: list[str] = []
		cur: ast.AST = node
		while isinstance(cur, ast.Attribute):
			parts.append(cur.attr)
			cur = cur.value
		if isinstance(cur, ast.Name):
			parts.append(cur.id)
			return ".".join(reversed(parts))
	return None


def _collect_model_infos(models_path: Path) -> dict[str, ModelInfo]:
	infos: dict[str, ModelInfo] = {}
	for build_file in models_path.glob("**/build.py"):
		with open(build_file, "r", encoding="utf-8") as f:
			tree = ast.parse(f.read(), filename=str(build_file))

		for node in tree.body:
			if not isinstance(node, ast.ClassDef):
				continue

			task_map: dict[str, list[str]] = {}
			model_callable_name: str | None = None
			doc_link: str | None = None
			for stmt in node.body:
				if not isinstance(stmt, ast.Assign):
					continue
				if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
					continue

				target_name = stmt.targets[0].id
				if target_name in {"task", "tasks"} and isinstance(stmt.value, ast.Dict):
					for k, v in zip(stmt.value.keys, stmt.value.values):
						if isinstance(k, ast.Constant) and isinstance(k.value, str):
							methods: list[str] = []
							if isinstance(v, ast.List):
								for method_node in v.elts:
									if isinstance(method_node, ast.Constant) and isinstance(method_node.value, str):
										methods.append(method_node.value)
							task_map[k.value] = methods

				if target_name == "model":
					model_callable_name = _expr_to_name(stmt.value)

				if target_name in {"doc_link", "dock_link"}:
					if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
						doc_link = stmt.value.value

				if target_name == "doc_links":
					if isinstance(stmt.value, ast.List):
						for link_node in stmt.value.elts:
							if isinstance(link_node, ast.Constant) and isinstance(link_node.value, str):
								doc_link = link_node.value
								break

			infos[node.name] = ModelInfo(
				class_name=node.name,
				model_callable_name=model_callable_name,
				task_map=task_map,
				docstring=inspect.cleandoc(ast.get_docstring(node) or ""),
				doc_link=doc_link,
				module_path=_module_path_for_file(build_file),
				source_path=build_file,
			)
	return infos


def _collect_factory_registry(kind: str) -> dict[str, type[Any]]:
	root = _losses_root() if kind == "loss" else _optimizers_root()
	attr_name = "loss" if kind == "loss" else "optimizer"
	registry: dict[str, type[Any]] = {}

	for file_path in root.glob("**/*.py"):
		if file_path.name == "__init__.py":
			continue

		module_path = _module_path_for_file(file_path)
		try:
			module = importlib.import_module(module_path)
		except (ImportError, ModuleNotFoundError, OSError) as exc:
			typer.secho(f"Skipping {module_path}: {exc}", fg=typer.colors.YELLOW, err=True)
			continue

		for _, cls in inspect.getmembers(module, inspect.isclass):
			if cls.__module__ != module.__name__:
				continue
			if getattr(cls, attr_name, None) is None:
				continue
			registry[cls.__name__] = cls

	return registry


def _collect_mode_callables(cls: type[Any]) -> list[tuple[str, Any]]:
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


def _callable_signature(module_path: str, class_name: str) -> inspect.Signature | None:
	module = importlib.import_module(module_path)
	cls = getattr(module, class_name)
	build_callable = getattr(cls, "build", None)
	if callable(build_callable):
		build_sig = inspect.signature(build_callable)
		build_params = tuple(build_sig.parameters.values())
		has_generic_var = any(
			p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
			for p in build_params
		)
		if not has_generic_var:
			return build_sig
	callable_obj = getattr(cls, "model", None)
	if callable_obj is None:
		callable_obj = getattr(cls, "loss", None)
	if callable_obj is None:
		callable_obj = getattr(cls, "optimizer", None)
	if callable_obj is None:
		raise AttributeError(f"Class '{class_name}' has no build/model/loss/optimizer callable")
	return inspect.signature(callable_obj)


def _format_annotation(annotation: Any) -> str:
	if annotation is inspect.Parameter.empty:
		return "Any"
	if isinstance(annotation, type):
		return annotation.__name__

	text = str(annotation)
	text = text.replace("typing.", "")
	text = text.replace("collections.abc.", "")
	return text


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


def build_doc_cli(*rows: RenderableType) -> Table:
	table = Table(show_header=False, box=box.SIMPLE_HEAVY, pad_edge=False, expand=False, show_lines=True)
	table.add_column("value")

	for row in rows:
		if isinstance(row, (Text, Table)):
			table.add_row(row)
		else:
			table.add_row(Text(str(row), style="white"))

	return table


def _build_model_meta_table(info: ModelInfo) -> Table:
	model_name = Text(info.class_name, style="bold cyan")

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
		signature = _callable_signature(info.module_path, info.class_name)
	except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError, OSError) as exc:
		signature = None
		arg_error = f"{type(exc).__name__}: {exc}"

	parameters_table = _build_arguments_table(signature, arg_error)
	return build_doc_cli(model_name, description_block, tasks_block, parameters_table)


def _build_arguments_table(signature: inspect.Signature | None, error_message: str | None = None) -> Table:
	table = Table(title="", header_style="bold bright_white")
	table.add_column("Parameter", style="yellow")
	table.add_column("Type", style="green")
	table.add_column("Default / Value", style="magenta")

	if signature is None:
		error_text = (
			f"unavailable ({error_message})"
			if error_message
			else "unavailable (signature lookup failed)"
		)
		table.add_row(
			"args",
			Text("-", style="white"),
			Text(error_text, style="yellow"),
		)
		return table

	for param in signature.parameters.values():
		name = param.name
		is_variadic = False
		if param.kind is inspect.Parameter.VAR_POSITIONAL:
			name = f"*{name}"
			is_variadic = True
		if param.kind is inspect.Parameter.VAR_KEYWORD:
			name = f"**{name}"
			is_variadic = True

		type_text = _highlight_type(_format_annotation(param.annotation))
		if param.default is inspect.Parameter.empty and not is_variadic:
			default_text: Text | str = Text("required", style="bright_yellow")
		elif param.default is inspect.Parameter.empty:
			default_text = "-"
		else:
			default_text = _format_default(param.default)

		table.add_row(Text(name, style="yellow"), type_text, default_text)

	return table


def _format_callable_header(factory_name: str, callable_name: str, callable_obj: Any) -> str:
	signature = inspect.signature(callable_obj)
	parts: list[str] = []
	for param in signature.parameters.values():
		if param.kind is inspect.Parameter.VAR_POSITIONAL:
			parts.append(f"*{param.name}")
		elif param.kind is inspect.Parameter.VAR_KEYWORD:
			parts.append(f"**{param.name}")
		else:
			parts.append(param.name)
	return f"{factory_name}.{callable_name}({', '.join(parts)})"


def _ast_to_text(node: ast.AST | None) -> str | None:
	if node is None:
		return None
	return ast.unparse(node)


def _class_init_signature_from_file(file_path: Path, class_name: str) -> str | None:
	if not file_path.exists():
		return None

	with open(file_path, "r", encoding="utf-8") as f:
		tree = ast.parse(f.read(), filename=str(file_path))

	class_node = next(
		(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name),
		None,
	)
	if class_node is None:
		return None

	init_node = next(
		(
			n
			for n in class_node.body
			if isinstance(n, ast.FunctionDef) and n.name == "__init__"
		),
		None,
	)
	if init_node is None:
		return "(self, *args, **kwargs)"

	args = init_node.args
	arg_nodes = args.args
	defaults = args.defaults
	default_offset = len(arg_nodes) - len(defaults)
	parts: list[str] = []

	for idx, arg in enumerate(arg_nodes):
		text = arg.arg
		ann = _ast_to_text(arg.annotation)
		if ann:
			text = f"{text}: {ann}"
		if idx >= default_offset:
			default_node = defaults[idx - default_offset]
			default_text = _ast_to_text(default_node) or "..."
			text = f"{text} = {default_text}"
		parts.append(text)

	if args.vararg:
		var_text = f"*{args.vararg.arg}"
		ann = _ast_to_text(args.vararg.annotation)
		if ann:
			var_text = f"{var_text}: {ann}"
		parts.append(var_text)

	for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
		kw_text = kw_arg.arg
		ann = _ast_to_text(kw_arg.annotation)
		if ann:
			kw_text = f"{kw_text}: {ann}"
		if kw_default is not None:
			kw_default_text = _ast_to_text(kw_default) or "..."
			kw_text = f"{kw_text} = {kw_default_text}"
		parts.append(kw_text)

	if args.kwarg:
		kw_text = f"**{args.kwarg.arg}"
		ann = _ast_to_text(args.kwarg.annotation)
		if ann:
			kw_text = f"{kw_text}: {ann}"
		parts.append(kw_text)

	return f"({', '.join(parts)})"


def _build_init_arguments_table_from_file(file_path: Path, class_name: str) -> Table:
	table = Table(title="", header_style="bold bright_white")
	table.add_column("Parameter", style="yellow")
	table.add_column("Type", style="green")
	table.add_column("Default / Value", style="magenta")

	if not file_path.exists():
		table.add_row(
			"args",
			Text("-", style="white"),
			Text("unavailable (template file not found)", style="yellow"),
		)
		return table

	with open(file_path, "r", encoding="utf-8") as f:
		tree = ast.parse(f.read(), filename=str(file_path))

	class_node = next(
		(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name),
		None,
	)
	if class_node is None:
		table.add_row(
			"args",
			Text("-", style="white"),
			Text("unavailable (class not found)", style="yellow"),
		)
		return table

	init_node = next(
		(
			n
			for n in class_node.body
			if isinstance(n, ast.FunctionDef) and n.name == "__init__"
		),
		None,
	)
	if init_node is None:
		table.add_row(
			"args",
			Text("-", style="white"),
			Text("unavailable (__init__ not found)", style="yellow"),
		)
		return table

	args = init_node.args
	arg_nodes = args.args
	defaults = args.defaults
	default_offset = len(arg_nodes) - len(defaults)

	for idx, arg in enumerate(arg_nodes):
		if arg.arg == "self":
			continue

		ann = _ast_to_text(arg.annotation) or "Any"
		type_text = _highlight_type(ann)
		if idx >= default_offset:
			default_node = defaults[idx - default_offset]
			default_text = _ast_to_text(default_node) or "..."
			default_value: Text | str = default_text
		else:
			default_value = Text("required", style="bright_yellow")

		table.add_row(Text(arg.arg, style="yellow"), type_text, default_value)

	if args.vararg:
		ann = _ast_to_text(args.vararg.annotation) or "Any"
		table.add_row(
			Text(f"*{args.vararg.arg}", style="yellow"),
			_highlight_type(ann),
			"-",
		)

	for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
		ann = _ast_to_text(kw_arg.annotation) or "Any"
		if kw_default is None:
			default_value = Text("required", style="bright_yellow")
		else:
			default_value = _ast_to_text(kw_default) or "..."
		table.add_row(Text(kw_arg.arg, style="yellow"), _highlight_type(ann), default_value)

	if args.kwarg:
		ann = _ast_to_text(args.kwarg.annotation) or "Any"
		table.add_row(
			Text(f"**{args.kwarg.arg}", style="yellow"),
			_highlight_type(ann),
			"-",
		)

	return table


def _class_docstring_from_file(file_path: Path, class_name: str) -> str | None:
	if not file_path.exists():
		return None

	with open(file_path, "r", encoding="utf-8") as f:
		tree = ast.parse(f.read(), filename=str(file_path))

	class_node = next(
		(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name),
		None,
	)
	if class_node is None:
		return None
	return inspect.cleandoc(ast.get_docstring(class_node) or "")


def _show_lightning_module_docs() -> None:
	template_file = (
		_templates_root() / "semantic_segmentation" / "supervised" / "src" / "lightning_module.py"
	)
	class_name = "GeosaveLightningModule"
	doc = _class_docstring_from_file(template_file, class_name)
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
		build_doc_cli(
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
	template_file = (
		_templates_root() / "semantic_segmentation" / "supervised" / "src" / "data_module.py"
	)
	class_name = "GeosaveDataModule"
	doc = _class_docstring_from_file(template_file, class_name)
	parameters_table = _build_init_arguments_table_from_file(template_file, class_name)

	global_keys_block = Text(
		"# Global settings\ndata:\n  data_dir:",
		style="green",
	)
	local_keys_block = Text(
		"# Local settings\ndata:\n  fit:\n    data_dir:",
		style="green",
	)
	desc_block = Text(doc.splitlines()[0] if doc else "No class docstring available.", style="white")

	console.print(
		build_doc_cli(
			Text("LightningDataModule", style="bold cyan"),
			desc_block,
			global_keys_block,
			local_keys_block,
			Text("Parameters are declared in your custom data module implementation (for example data_modules.py).", style="white"),
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
				build_doc_cli(
					Text("Trainer", style="bold cyan"),
					Text("Unable to import lightning Trainer in current environment.", style="yellow"),
				)
			)
			return

	trainer_doc = inspect.getdoc(Trainer) or "PyTorch Lightning Trainer configuration."
	trainer_doc_short = trainer_doc.split("\n\n", maxsplit=1)[0]
	trainer_sig_table = _build_arguments_table(inspect.signature(Trainer))

	global_keys_block = Text(
		"# Global settings\ntrainer:\n  max_epochs:\n  callbacks:\n  logger:",
		style="green",
	)
	local_keys_block = Text(
		"# Local settings\nfit:\n  trainer:\ntest:\n  trainer:\npredict:\n  trainer:",
		style="green",
	)

	console.print(
		build_doc_cli(
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
	registry = _collect_factory_registry(kind)

	if not registry:
		console.print(
			build_doc_cli(
				Text(title, style="bold cyan"),
				Text("No factory classes discovered in current environment.", style="yellow"),
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

	resolved = _resolve_from_choices(selected_name, names)
	if not resolved:
		typer.secho(f"Unknown {kind}: {selected_name}", fg=typer.colors.RED, err=True)
		raise typer.Exit(1)

	cls = registry[resolved]
	class_doc = inspect.getdoc(cls) or "No class docstring available."
	class_doc_short = class_doc.split("\n\n", maxsplit=1)[0]

	target_attr = "loss" if kind == "loss" else "optimizer"
	target_callable = getattr(cls, target_attr)
	target_sig_table = _build_arguments_table(inspect.signature(target_callable))

	mode_rows: list[RenderableType] = []
	for mode_name, mode_callable in _collect_mode_callables(cls):
		mode_sig_table = _build_arguments_table(inspect.signature(mode_callable))
		mode_doc = inspect.getdoc(mode_callable) or "No mode docstring available."
		mode_doc_short = mode_doc.split("\n\n", maxsplit=1)[0]
		mode_header = _format_callable_header(cls.__name__, mode_name, mode_callable)
		mode_rows.extend(
			[
				Text(mode_header, style="bold green"),
				Text(mode_doc_short, style="white"),
				mode_sig_table,
			]
		)

	doc_links = getattr(cls, "doc_links", None)
	doc_link_text = Text(doc_links[0], style="blue underline") if isinstance(doc_links, list) and doc_links else Text("https://docs.com", style="blue underline")

	console.print(
		build_doc_cli(
			Text(cls.__name__, style="bold cyan"),
			Text(class_doc_short, style="white"),
			doc_link_text,
			Text(f"base {target_attr} signature", style="bold yellow"),
			target_sig_table,
			*mode_rows,
		)
	)


def _show_model_docs(task_arg: str | None, method_arg: str | None, model_arg: str | None) -> None:
	templates_path = _templates_root()
	models_path = _models_root()
	task_map = get_tasks(templates_path)
	info_map = _collect_model_infos(models_path)

	tasks = sorted(task_map.keys())
	task = None
	if task_arg:
		task = _resolve_from_choices(task_arg, tasks)
		if not task:
			typer.secho(f"Unknown task: {task_arg}", fg=typer.colors.RED, err=True)
			raise typer.Exit(1)
	else:
		task = questionary.select(
			"Choose task:", choices=[questionary.Choice(t, value=t) for t in tasks]
		).ask()
		if not task:
			raise typer.Exit(1)

	methods = task_map[task]
	method = ""
	if methods:
		if method_arg:
			resolved = _resolve_from_choices(method_arg, methods)
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
				"Choose method:", choices=[questionary.Choice(m, value=m) for m in methods]
			).ask()
			if not method:
				raise typer.Exit(1)

	model_names = sorted(
		name
		for name, info in info_map.items()
		if task in info.task_map and (not method or len(info.task_map[task]) == 0 or method in info.task_map[task])
	)
	if not model_names:
		typer.secho(
			f"No models available for task='{task}' method='{method or 'all'}'",
			fg=typer.colors.YELLOW,
		)
		return

	model_name = None
	if model_arg:
		model_name = _resolve_from_choices(model_arg, model_names)
		if not model_name:
			typer.secho(f"Unknown model: {model_arg}", fg=typer.colors.RED, err=True)
			raise typer.Exit(1)
	else:
		model_name = questionary.select(
			"Choose model:", choices=[questionary.Choice(m, value=m) for m in model_names]
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
		selection = questionary.select(
			"Choose Geosave docs:",
			choices=[
				questionary.Choice("LightningModule", value="lightningmodule"),
				questionary.Choice("LightningDataModule", value="datamodule"),
				questionary.Choice("Trainer", value="trainer"),
				questionary.Choice("Models", value="model"),
				questionary.Choice("Losses", value="loss"),
				questionary.Choice("Optimizers", value="optimizer"),
			],
		).ask()
		if not selection:
			raise typer.Exit(1)
		section = selection

	section_norm = _normalize_slug(section)
	if section_norm in {"lightningmodule", "lightning module"}:
		_show_lightning_module_docs()
		return

	if section_norm in {"datamodule", "data module"}:
		_show_data_module_docs()
		return

	if section_norm in {"trainer"}:
		_show_trainer_docs()
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
		"Valid sections: lightningmodule, datamodule, trainer, model, loss, optimizer",
		fg=typer.colors.YELLOW,
		err=True,
	)
	raise typer.Exit(1)
