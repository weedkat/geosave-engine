from __future__ import annotations
import typer
from pathlib import Path

def find_configs(directory: Path) -> list[Path]:
    configs_dir = directory / "configs"
    if configs_dir.exists() and configs_dir.is_dir():
        return list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    return []

def find_artifact_parents(directory: Path) -> list[Path]:
    artifacts_path = directory / "artifacts"
    choices = []
    if artifacts_path.exists() and artifacts_path.is_dir():
        for d in artifacts_path.iterdir():
            if d.is_dir() and (list(d.glob("*.yaml")) or list(d.glob("*.yml"))):
                choices.append(d)
    return choices

def find_script(directory: Path, script_names: str | list[str]) -> Path:
    if isinstance(script_names, str):
        script_names = [script_names]
        
    for name in script_names:
        scripts = list(directory.rglob(name))
        if scripts:
            return scripts[0]
            
    typer.secho(f"Error: {script_names[0]} not found in {directory}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)
