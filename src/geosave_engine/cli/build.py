from __future__ import annotations
import questionary
import os
import typer
from pathlib import Path

from geosave_engine.cli.utils.parse import get_model_list, tasks
from geosave_engine.cli.utils.generate import generate_project

def build_project(name, dir: str):
    """
    Build a new GeoSave project workspace.
    
    This command interactively scaffolds a new project by asking you to select an AI task, 
    a training method, and the specific models you want to use. It copies the necessary 
    templates and generates a ready-to-use workspace with a tracking `geosave.toml` file.
    """
    typer.secho("Building the project...", fg=typer.colors.CYAN)
    
    if name is None:
        name = questionary.text(
            "Enter the name of the build:",
            validate=lambda text: bool(text.strip()) or "Name cannot be empty"
        ).ask()
        if not name:
            raise typer.Exit()

    task = questionary.select(
        "Select the AI task:",
        choices=[t for t in tasks]
    ).ask()
    
    if not task:
        raise typer.Exit()
    
    methods = tasks.get(task)
    if not methods:
        typer.secho(f"Error: No methods/templates found for the task '{task}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    method = questionary.select(
        "Select the methods:",
        choices=[m for m in methods]
    ).ask()

    if not method or method not in methods:
        typer.secho(f"Error: Method '{method}' does not exist for task '{task}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    model_choices = get_model_list(task, method)

    if not model_choices:
        typer.secho("Error: No models found for the selected task and method.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    models = questionary.checkbox(
        "Select the models (at least one):",
        choices=model_choices
    ).ask()

    # checkbox returns an empty list if nothing selected, None if cancelled
    if models is None:
        raise typer.Exit()
    if not models:
        typer.secho("Error: You must select at least one model.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    description = questionary.text(
        "Enter the description of the build:",
        default="A GeoSave Engine project.",
    ).ask()

    if description is None:
        raise typer.Exit()

    if generate_project(dir, name, task, method, models, description):
        # Inject selected model into config file (pick the first model)
        config_path = Path(dir) / name / "configs" / "default.yaml"
        if config_path.exists():
            import yaml
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                if not config_data:
                    config_data = {}
                if "model" not in config_data or not isinstance(config_data["model"], dict):
                    config_data["model"] = {}
                config_data["model"]["name"] = models[0]
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config_data, f, sort_keys=False)
                typer.secho(f"Injected selected model '{models[0]}' into config file.", fg=typer.colors.BLUE)
            except Exception as e:
                typer.secho(f"Warning: Could not update config file with selected model: {e}", fg=typer.colors.YELLOW)
        typer.secho(f"Project '{name}' created successfully at {os.path.join(dir, name)}", fg=typer.colors.GREEN)
