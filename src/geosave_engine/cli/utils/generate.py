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
            available_models = discover_available_models(task, method)
        except Exception as e:
            typer.secho(f"An error occurred during model discovery: {e}", fg=typer.colors.RED, err=True)
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
                if item.targets[0].id != "tasks":
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


def _flatten_registry(imports_by_module: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for module_name in sorted(imports_by_module.keys()):
        entries.extend(imports_by_module[module_name])
    return entries


def discover_available_models(task: str, method: str) -> list[str]:
    package_root = Path(__file__).parent.parent.parent / "models"
    model_imports = _collect_model_factories(package_root, task, method)
    if not model_imports:
        raise ValueError(f"No model factories found for task='{task}' and method='{method}'.")

    model_entries = _flatten_registry(model_imports)
    return [key for _, key in model_entries]


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
    print(discover_available_models("semantic segmentation", "supervised"))