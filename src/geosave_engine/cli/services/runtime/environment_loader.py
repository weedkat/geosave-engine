from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer


class EnvironmentLoader:
    @staticmethod
    def load(project_dir: Path, current_dir: Path) -> dict[str, str]:
        env = os.environ.copy()

        search_paths = [
            project_dir / ".env",
            project_dir / ".env.example",
            current_dir / ".env",
            current_dir / ".env.example",
            current_dir.parent / ".env",
            current_dir.parent / ".env.example",
        ]

        env_file = None
        for path in search_paths:
            if path.exists() and path.is_file() and path.name == ".env":
                env_file = path
                break

        if env_file is None:
            for path in search_paths:
                if path.exists() and path.is_file() and path.name == ".env.example":
                    new_env = path.parent / ".env"
                    try:
                        shutil.copy2(path, new_env)
                        typer.secho(
                            f"Copied {path} to {new_env} as .env was missing.",
                            fg=typer.colors.YELLOW,
                        )
                        env_file = new_env
                        break
                    except Exception as error:
                        typer.secho(
                            f"Error copying {path} to {new_env}: {error}",
                            fg=typer.colors.RED,
                            err=True,
                        )
                        raise typer.Exit(1)

        if env_file is None:
            typer.secho(
                "Error: No .env or .env.example file found in "
                f"{project_dir}, {current_dir}, or {current_dir.parent}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        try:
            with open(env_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env[key.strip()] = value.strip().strip("'\"")
        except Exception as error:
            typer.secho(
                f"Warning: Failed to load .env file at {env_file}: {error}",
                fg=typer.colors.YELLOW,
            )

        env["GEOSAVE_PROJECT_DIR"] = str(project_dir.resolve())
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{project_dir.resolve()}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(project_dir.resolve())

        return env
