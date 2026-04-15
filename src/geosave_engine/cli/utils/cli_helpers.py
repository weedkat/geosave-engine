from __future__ import annotations
from pathlib import Path
import questionary
import typer

from geosave_engine.cli.utils.parse import validate_workspace
from geosave_engine.cli.utils.search import (
    find_configs,
    find_artifact_parents,
)


def validate_workspace_with_feedback(project_dir: Path) -> dict:
    """
    Validate workspace and display feedback to the user.
    
    Args:
        project_dir: Path to the GeoSave project directory
        
    Returns:
        Workspace configuration dictionary
    """
    workspace_config = validate_workspace(project_dir)

    project_name = workspace_config.get("project_name", "Unknown Project")
    if isinstance(project_name, tuple):
        project_name = project_name[0]

    typer.secho(
        f"Found GeoSave project workspace: '{project_name}'", fg=typer.colors.CYAN
    )

    return workspace_config


def get_config_args(project_dir: Path, config: str | None) -> list[str]:
    """
    Get configuration file arguments, prompting user if needed.
    
    Args:
        project_dir: Path to the GeoSave project directory
        config: Optional path to configuration file
        
    Returns:
        List of arguments to pass to the training script
    """
    if config:
        return ["--config", config]

    config_files = find_configs(project_dir)
    if config_files:
        choices = [
            questionary.Choice(f.name, value=str(f.resolve())) for f in config_files
        ]
        selected = questionary.select(
            "Select the configuration file:", choices=choices
        ).ask()
        if selected:
            return ["--config", selected]
        raise typer.Exit(1)

    typer.secho(
        f"Error: No .yaml or .yml configuration files found in '{project_dir / 'configs'}'.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(1)


def get_artifacts_args(project_dir: Path, artifacts: str | None) -> list[str]:
    """
    Get artifact directory arguments, prompting user if needed.
    
    Args:
        project_dir: Path to the GeoSave project directory
        artifacts: Optional path to artifacts directory
        
    Returns:
        List of arguments to pass to the test/inference script
    """
    if artifacts:
        return ["--model", artifacts]

    artifact_dirs = find_artifact_parents(project_dir)
    if artifact_dirs:
        choices = [
            questionary.Choice(d.name, value=str(d.resolve())) for d in artifact_dirs
        ]
        selected = questionary.select(
            "Select the model artifacts containing config:", choices=choices
        ).ask()
        if selected:
            return ["--model", selected]
        raise typer.Exit(1)

    typer.secho(
        f"Error: No valid artifact directories containing config files found in '{project_dir / 'artifacts'}'.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(1)
