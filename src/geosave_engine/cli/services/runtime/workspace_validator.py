from __future__ import annotations

from pathlib import Path

import toml
import typer


class WorkspaceValidator:
    @staticmethod
    def validate(project_dir: Path) -> dict:
        config_path = project_dir / "geosave.toml"

        if not config_path.exists():
            typer.secho(
                f"Error: GeoSave workspace not found. Could not find {config_path}",
                fg=typer.colors.RED,
                err=True,
            )
            typer.secho(
                "Make sure you are in a directory created by 'geosave-engine build' "
                "or pass the correct path.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(1)

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = toml.load(handle)
        except Exception as error:
            typer.secho(
                f"Error reading geosave.toml: {error}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        return config

    @staticmethod
    def validate_with_feedback(project_dir: Path) -> dict:
        workspace_config = WorkspaceValidator.validate(project_dir)

        project_name = workspace_config.get("project_name", "Unknown Project")
        if isinstance(project_name, tuple):
            project_name = project_name[0]

        typer.secho(
            f"Found GeoSave project workspace: '{project_name}'",
            fg=typer.colors.CYAN,
        )

        return workspace_config
