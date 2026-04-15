from __future__ import annotations
import os
import shutil
import ast
from pathlib import Path
import toml
import typer


def generate_project(
    dir: str,
    name: str,
    task: str,
    method: str,
    selected_model: str,
    description: str,
    template_dir: Path,
) -> bool:
    project_template_dir = template_dir / task.replace(" ", "_") / method.replace(" ", "_")

    try:
        if copier(str(project_template_dir), os.path.join(dir, name)):
            os.makedirs(os.path.join(dir, name, "data"), exist_ok=True)
            os.makedirs(os.path.join(dir, name, "artifacts"), exist_ok=True)
            main_template_path = os.path.join(template_dir, "main_template.py")
            main_py_path = os.path.join(dir, name, "main.py")
            copier(str(main_template_path), main_py_path)
        else:
            return False

        try:
            factory_path = os.path.join(dir, name, "src", "factories.py")
            available_models = inject_model_factory(task, method, factory_path)
        except Exception as e:
            typer.secho(f"An error occurred during factory file generation: {e}", fg=typer.colors.RED, err=True)
            return False

    except Exception as e:
        typer.secho(f"An error occurred during copying: {e}", fg=typer.colors.RED, err=True)
        return False

    env_path = template_dir / ".env"
    if env_path.exists():
        copier(str(env_path), os.path.join(dir, name, ".env"))

    with open(os.path.join(dir, name, "geosave.toml"), "w", encoding="utf-8") as f:
        toml.dump({
            "project_name": name,
            "task": task,
            "method": method,
            "models": available_models,
            "selected_model": selected_model,
            "description": description
        }, f)
        
    return True


def _module_path_for_file(file_path: Path, package_root: Path) -> str:
    rel_path = file_path.relative_to(package_root.parent.parent)
    return str(rel_path.with_suffix("")).replace(os.sep, ".")


def _collect_model_factories(package_root: Path, task: str, method: str) -> dict[str, list[tuple[str, str]]]:
    imports_by_module: dict[str, list[tuple[str, str]]] = {}

    for file in package_root.glob("**/build.py"):
        if file.name == "__init__.py":
            continue

        with open(file, "r", encoding="utf-8") as f:
            try:
                node = ast.parse(f.read())
            except SyntaxError:
                continue

        for class_node in node.body:
            if not isinstance(class_node, ast.ClassDef):
                continue

            task_map: dict[str, list[str]] = {}
            for item in class_node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                    continue
                if item.targets[0].id not in {"task", "tasks"}:
                    continue
                if not isinstance(item.value, ast.Dict):
                    continue
                for key_node, value_node in zip(item.value.keys, item.value.values):
                    if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                        continue
                    methods: list[str] = []
                    if isinstance(value_node, ast.List):
                        methods = [
                            elt.value
                            for elt in value_node.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                    task_map[key_node.value] = methods

            if task not in task_map:
                continue
            if task_map[task] and method not in task_map[task]:
                continue

            module_path = _module_path_for_file(file, package_root)
            model_name = class_node.name
            model_key = class_node.name
            if not any(existing_name == class_node.name for existing_name, _ in imports_by_module.get(module_path, [])):
                imports_by_module.setdefault(module_path, []).append((model_name, model_key))

    return imports_by_module


def _render_registry_lines(registry_name: str, registry_entries: list[tuple[str, str]]) -> list[str]:
    lines = [f"{registry_name} = {{"]
    for class_name, key in registry_entries:
        lines.append(f"    '{key}': {class_name},")
    lines.append("}")
    return lines


def _flatten_registry(imports_by_module: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for module_name in sorted(imports_by_module.keys()):
        entries.extend(imports_by_module[module_name])
    return entries


def _build_model_registry_source(task: str, method: str, package_root: Path) -> tuple[list[str], list[str]]:
    model_imports = _collect_model_factories(package_root, task, method)
    if not model_imports:
        raise ValueError(f"No model factories found for task='{task}' and method='{method}'.")

    lines: list[str] = []
    for modname in sorted(model_imports.keys()):
        cls_names = [info[0] for info in model_imports[modname]]
        lines.append(f"from {modname} import {', '.join(cls_names)}")

    model_entries = _flatten_registry(model_imports)
    lines.extend([
        "",
        *_render_registry_lines("MODEL_FACTORY", model_entries),
    ])

    return lines, [key for _, key in model_entries]


def _build_common_factory_source() -> list[str]:
    return [
        "_RAW_OPTIM_FACTORY = globals().get('OPTIM_FACTORY', []) or []",
        "_RAW_LOSS_FACTORY = globals().get('LOSS_FACTORY', []) or []",
        "",
        "OPTIMIZER_FACTORY = {cls.__name__: cls for cls in _RAW_OPTIM_FACTORY}",
        "LOSS_FACTORY = {cls.__name__: cls for cls in _RAW_LOSS_FACTORY}",
        "",
        "",
        "def _available_methods(factory_cls):",
        "    return sorted([name for name, value in factory_cls.__dict__.items() if isinstance(value, classmethod) and not name.startswith('_')])",
        "",
        "",
        "def _resolve_factory_callable(registry, kind, name, method=None, default_method=None):",
        "    factory_cls = registry.get(name)",
        "    if factory_cls is None:",
        "        raise ValueError(f\"Unknown {kind} '{name}'. Available: {', '.join(sorted(registry.keys()))}\")",
        "    if method is not None:",
        "        candidate = getattr(factory_cls, method, None)",
        "        if callable(candidate):",
        "            return candidate",
        "        raise ValueError(f\"{kind.capitalize()} '{name}' does not support method '{method}'. Available methods: {_available_methods(factory_cls)}\")",
        "    if default_method is not None:",
        "        candidate = getattr(factory_cls, default_method, None)",
        "        if callable(candidate):",
        "            return candidate",
        "    build_candidate = getattr(factory_cls, 'build', None)",
        "    if callable(build_candidate):",
        "        return build_candidate",
        "    raise ValueError(f\"{kind.capitalize()} '{name}' has no build method. Available methods: {_available_methods(factory_cls)}\")",
        "",
        "",
        "def build_loss(name, *args, method='full', **kwargs):",
        "    factory_callable = _resolve_factory_callable(LOSS_FACTORY, 'loss', name, method=method, default_method='full')",
        "    return factory_callable(*args, **kwargs)",
        "",
        "",
        "def build_optimizer(name, *args, method='full', **kwargs):",
        "    factory_callable = _resolve_factory_callable(OPTIMIZER_FACTORY, 'optimizer', name, method=method, default_method='full')",
        "    return factory_callable(*args, **kwargs)",
        "",
    ]


def inject_model_factory(task: str, method: str, output_path: str) -> list[str]:
    """Append a generated factory block to the template-owned factories.py file."""
    package_root = Path(__file__).parent.parent.parent / "models"
    common_lines = _build_common_factory_source()
    model_lines, model_keys = _build_model_registry_source(task, method, package_root)

    with open(output_path, "r", encoding="utf-8") as f:
        template_source = f.read()

    block_start = "# ==== GEOSAVE AUTO-GENERATED FACTORY BLOCK: START ===="
    block_end = "# ==== GEOSAVE AUTO-GENERATED FACTORY BLOCK: END ===="

    rendered_base = template_source
    if block_start in template_source:
        rendered_base = template_source.split(block_start, maxsplit=1)[0].rstrip()

    generated_lines = [
        block_start,
        "",
        *common_lines,
        *model_lines,
        "",
        "def build_model(name, *args, method=None, **kwargs):",
        "    factory_callable = _resolve_factory_callable(MODEL_FACTORY, 'model', name, method=method)",
        "    return factory_callable(*args, **kwargs)",
        "",
        block_end,
    ]

    rendered = rendered_base + "\n\n" + "\n".join(generated_lines) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    return model_keys


def copier(source: str, destination: str) -> bool:
    """
    Copies a file or directory recursively from the source path to the destination path.

    Args:
        source (str): The path to the source file or directory.
        destination (str): The path to the destination file or directory.

    Raises:
        FileNotFoundError: If the source path does not exist.
        IOError: If there is an error during copying.
    """

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    if os.path.exists(destination):
        import questionary
        choice = questionary.confirm(
            f"Warning: Destination '{destination}' already exists. Overwrite?"
        ).ask()
        
        if not choice:
            typer.secho("Copy operation cancelled.", fg=typer.colors.YELLOW)
            return False

    try:
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
            # Not printing here to avoid noise during project build. You can log it if needed
        else:
            shutil.copy2(source, destination)
    except IOError as e:
        raise IOError(f"Error copying from '{source}' to '{destination}': {e}")

    return True

if __name__ == "__main__":
    # Example usage:
    # source_path = "src/geosave_engine/core"
    # destination_path = "test"
    # copier(source_path, destination_path)
    inject_model_factory("semantic segmentation", "supervised", "factories.py")