from __future__ import annotations
import typer
import subprocess
import os
from pathlib import Path

def load_env(project_dir: Path, current_dir: Path) -> dict[str, str]:
    env = os.environ.copy()

    # Search order for .env files
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
    # If no .env, but .env.example exists, copy it to .env and use it
    if env_file is None:
        for path in search_paths:
            if path.exists() and path.is_file() and path.name == ".env.example":
                # Copy .env.example to .env in the same directory
                new_env = path.parent / ".env"
                try:
                    import shutil
                    shutil.copy2(path, new_env)
                    typer.secho(f"Copied {path} to {new_env} as .env was missing.", fg=typer.colors.YELLOW)
                    env_file = new_env
                    break
                except Exception as e:
                    typer.secho(f"Error copying {path} to {new_env}: {e}", fg=typer.colors.RED, err=True)
                    raise typer.Exit(1)

    if env_file is None:
        typer.secho(f"Error: No .env or .env.example file found in {project_dir}, {current_dir}, or {current_dir.parent}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip("'\"")
    except Exception as e:
        typer.secho(f"Warning: Failed to load .env file at {env_file}: {e}", fg=typer.colors.YELLOW)

    env["GEOSAVE_PROJECT_DIR"] = str(project_dir.resolve())

    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_dir.resolve()}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(project_dir.resolve())

    return env

def execute_script(project_name: str, task_name: str, script_path: Path, project_dir: Path, current_dir: Path, run_args: list[str], operation: str = "pipeline"):
    cmd_env = load_env(project_dir, current_dir)
    args_str = " ".join(run_args) if run_args else "No extra args"
    script_rel_path = script_path.relative_to(project_dir)
    
    task_display = f" (Task: {task_name})" if task_name else ""
    typer.secho(f"Starting {operation} for '{project_name}'{task_display}\nExecuting: `python {script_rel_path} {args_str}`", fg=typer.colors.GREEN)
    
    try:
        subprocess.run(["python", str(script_path.resolve())] + run_args, check=True, cwd=project_dir, env=cmd_env)
    except subprocess.CalledProcessError as e:
        typer.secho(f"{operation.capitalize()} failed with exit code {e.returncode}", fg=typer.colors.RED, err=True)
        raise typer.Exit(e.returncode)