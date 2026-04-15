from __future__ import annotations
import os
import shutil
import ast
from pathlib import Path
from collections import defaultdict
import toml
import typer

def generate_project(dir: str, name: str, task: str, method: str, models: list[str], description: str, template_dir: Path) -> bool:
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
            generate_models_file(models, os.path.join(dir, name, "src", "model_factory.py"))
        except Exception as e:
            typer.secho(f"An error occurred during model file generation: {e}", fg=typer.colors.RED, err=True)
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
            "models": models,
            "description": description
        }, f)
        
    return True

def generate_models_file(models: list[str], output_path: str) -> None:
    """
    Generates a Python file exporting a factory dictionary mapping model names to their classes.
    Scans the src/geosave_engine/models directory to find where the requested models are defined.
    """
    package_path = Path(__file__).parent.parent.parent / "models"
    
    # Store {"module.path.to.import": [("ModelClass", "model_key")]}
    imports_by_module = defaultdict(list)

    # Scan the AST of all build.py files in the models directory to find the imports
    for file in package_path.glob("**/build.py"):
        if file.name == "__init__.py":
            continue

        with open(file, "r", encoding="utf-8") as f:
            try:
                node = ast.parse(f.read())
            except SyntaxError:
                continue

            for n in node.body:
                if isinstance(n, ast.ClassDef) and n.name in models:
                    # Construct the module import path based relative to the models directory
                    rel_path = file.relative_to(package_path.parent.parent)
                    module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
                    
                    # Extract the model name attribute from AST, or use class name lowercased
                    model_key = n.name.lower()
                    for item in n.body:
                        if isinstance(item, ast.Assign) and len(item.targets) == 1:
                            target = item.targets[0]
                            if isinstance(target, ast.Name) and target.id == "name":
                                if isinstance(item.value, ast.Constant):
                                    model_key = item.value.value

                    # Ensure we do not add duplicates
                    if not any(cls_name == n.name for cls_name, _ in imports_by_module[module_path]):
                        imports_by_module[module_path].append((n.name, model_key))

    # Generate the file contents
    lines = []
    
    for modname, cls_info in imports_by_module.items():
        cls_names = [info[0] for info in cls_info]
        lines.append(f"from {modname} import {', '.join(cls_names)}")
        
    lines.append("\nfactory = {")
    for modname, cls_info in imports_by_module.items():
        for cls_name, model_key in cls_info:
            lines.append(f"    '{model_key}': {cls_name},")
    lines.append("}\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
    generate_models_file(["Unet"], "factory.py")