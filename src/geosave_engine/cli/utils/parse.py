from __future__ import annotations
import ast
from pathlib import Path
import typer
import toml


def get_tasks(template_path: Path) -> dict[str, list[str]]:
    tasks: dict[str, list[str]] = {}
    folders = [p for p in template_path.iterdir() if p.is_dir() and p.name != "__pycache__"]
    for folder in folders:
        methods = [p for p in folder.iterdir() if p.is_dir() and p.name != "__pycache__"]
        task_name = folder.name.replace("_", " ").lower()
        method_names = []
        for method in methods:
            method_name = method.name.replace("_", " ").lower()
            method_names.append(method_name)
        tasks[task_name] = method_names
    return tasks


def get_model_list(task: str, method: str, package_path: Path) -> list[str]:
    class_names = []
    for file in package_path.glob("**/build.py"):
        if file.name == "__init__.py":
            continue

        with open(file, "r", encoding="utf-8") as f:
            node = ast.parse(f.read())
            for n in node.body:
                if isinstance(n, ast.ClassDef):
                    for item in n.body:
                        # Look for class assignments
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                # Find where variable name is "task"
                                if isinstance(target, ast.Name) and target.id == "task":
                                    # Expecting task to be a dictionary
                                    if isinstance(item.value, ast.Dict):
                                        # Iterate pairs of keys and values
                                        for k, v in zip(item.value.keys, item.value.values):
                                            # Check if the dictionary key matches the task string
                                            if isinstance(k, ast.Constant) and k.value == task:
                                                # Check if the dictionary value is a list
                                                if isinstance(v, ast.List):
                                                    # Extract method strings from the list
                                                    methods = [elt.value for elt in v.elts if isinstance(elt, ast.Constant)]
                                                    # Match if no method requested, matching method found, or list is empty (supports all)
                                                    if not method or len(methods) == 0 or method in methods:
                                                        class_names.append(n.name)

    return class_names

def validate_workspace(project_dir: Path) -> dict:
    """Helper to detect geosave.toml and return the config."""
    config_path = project_dir / "geosave.toml"
    
    if not config_path.exists():
        typer.secho(f"Error: GeoSave workspace not found. Could not find {config_path}", fg=typer.colors.RED, err=True)
        typer.secho("Make sure you are in a directory created by 'geosave-engine build' or pass the correct path.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
    except Exception as e:
        typer.secho(f"Error reading geosave.toml: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
            
    return config

if __name__ == "__main__":
    pass
