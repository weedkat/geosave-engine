from __future__ import annotations

import shlex

import questionary
import typer

from geosave_engine.cli.services.search.project_search import ProjectSearch


class ArgumentSelector:
    def __init__(self, project_search: ProjectSearch) -> None:
        self.project_search = project_search

    def config_args(self, config: str | None) -> list[str]:
        project_dir = self.project_search.get_project_dir()
        if config:
            return ["--config", config]

        config_files = self.project_search.find_configs()
        if config_files:
            choices = [
                questionary.Choice(config_file.name, value=str(config_file.resolve()))
                for config_file in config_files
            ]
            selected = questionary.select(
                "Select the configuration file:",
                choices=choices,
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

    def artifact_args(self, artifacts: str | None) -> list[str]:
        project_dir = self.project_search.get_project_dir()
        if artifacts:
            return ["--model", artifacts]

        artifact_dirs = self.project_search.find_artifact_parents()
        if artifact_dirs:
            choices = [
                questionary.Choice(directory.name, value=str(directory.resolve()))
                for directory in artifact_dirs
            ]
            selected = questionary.select(
                "Select the model artifacts containing config:",
                choices=choices,
            ).ask()
            if selected:
                return ["--model", selected]
            raise typer.Exit(1)

        typer.secho(
            "Error: No valid artifact directories containing config files found in "
            f"'{project_dir / 'artifacts'}'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    def resolve_script_and_args(
        self,
        script_name: str | None,
        extra_args: list[str],
    ) -> tuple[str, list[str]]:
        if script_name:
            return script_name, extra_args

        if extra_args and not extra_args[0].startswith("-"):
            return extra_args[0], extra_args[1:]

        project_dir = self.project_search.get_project_dir()
        scripts = self.project_search.list_scripts_in_scripts_dir()
        scripts_dir = project_dir / "scripts"
        choices = [
            questionary.Choice(
                str(script_path.relative_to(scripts_dir)),
                value=str(script_path.relative_to(scripts_dir)),
            )
            for script_path in scripts
        ]

        selected = questionary.select(
            "Select the script to run:",
            choices=choices,
        ).ask()
        if selected:
            return selected, extra_args
        raise typer.Exit(1)

    def script_args(self, extra_args: list[str]) -> list[str]:
        if extra_args:
            return extra_args

        raw_args = questionary.text(
            "Enter additional script arguments (optional):",
            default="",
        ).ask()
        if raw_args is None:
            raise typer.Exit(1)

        text = raw_args.strip()
        if not text:
            return []

        try:
            return shlex.split(text)
        except ValueError as error:
            typer.secho(
                f"Error: Invalid argument string: {error}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
