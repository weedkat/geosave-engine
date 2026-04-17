from __future__ import annotations

import subprocess
from pathlib import Path

import typer


class ScriptExecutor:
    @staticmethod
    def run(script_path: Path, args: list[str], cwd: Path, env: dict[str, str]) -> None:
        try:
            subprocess.run(
                ["python", str(script_path.resolve())] + args,
                check=True,
                cwd=cwd,
                env=env,
            )
        except subprocess.CalledProcessError as error:
            typer.secho(
                f"Script failed with exit code {error.returncode}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(error.returncode)
