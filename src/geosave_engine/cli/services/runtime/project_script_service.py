from __future__ import annotations

from pathlib import Path

import typer

from geosave_engine.cli.services.runtime.environment_loader import (
    EnvironmentLoader,
)
from geosave_engine.cli.services.runtime.script_executor import ScriptExecutor


class ProjectScriptService:
    @staticmethod
    def execute(
        project_name: str,
        task_name: str,
        script_path: Path,
        project_dir: Path,
        current_dir: Path,
        run_args: list[str],
        operation: str,
    ) -> None:
        cmd_env = EnvironmentLoader.load(project_dir, current_dir)
        args_str = " ".join(run_args) if run_args else "No extra args"
        script_rel_path = script_path.relative_to(project_dir)
        task_display = f" (Task: {task_name})" if task_name else ""
        typer.secho(
            f"Starting {operation} for '{project_name}'{task_display}\n"
            f"Executing: `python {script_rel_path} {args_str}`",
            fg=typer.colors.GREEN,
        )
        ScriptExecutor.run(script_path, run_args, project_dir, cmd_env)
