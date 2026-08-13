from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
import questionary as qu

from ..core.templates import get_tasks, task_dir
from ..core.workspace import create_workspace
from ..core.toml import create_toml
from ..core.prompts import prompt_required_text, prompt_optional_text, prompt_select

BASE_TASK = "blank"

def make_choice(title: str, value: str | None, description: str) -> qu.Choice:
    return qu.Choice(
        title=title,
        value=value,
        description=description,
    )

def create(
    name: Annotated[
        Optional[str],
        typer.Argument(
            exists=False,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
            help="Project name to create the workspace in.",
        ),
    ] = None,
    description: Annotated[
            Optional[str],
            typer.Option(
                "-d",
                "--description",
                help="A brief description of the workspace.",
            ),
        ] = None,
    task: Annotated[
        Optional[str],
        typer.Option(
            "-t",
            "--task",
            help="The task for the workspace.",
            ),
        ] = None,
    method: Annotated[
        Optional[str],
        typer.Option(
            "-m",
            "--method",
            help="The method for the workspace.",
        ),
    ] = None,
) -> None:
    """Create one GeoSave workspace."""

    method_templates = get_tasks()

    if name is None:
        name = prompt_required_text("Enter a name for the workspace:")

    if description is None:
        description = prompt_optional_text("Enter a description for the workspace (optional):")
    
    if not task:
        choices = [make_choice(BASE_TASK, None, "No task selected.")]
        for task in method_templates:
            file_txt = (task_dir() / task / "description.txt")

            if file_txt.exists():
                description = file_txt.read_text().strip()
            else:
                description = "No description available."

            choices.append(make_choice(task, task, description))

        task = prompt_select(
            "Select a task for the workspace:",
            choices=choices,
        )

    if not method and task != BASE_TASK:
        choices = []
        for method in method_templates[task]:
            file_txt = (task_dir() / task / method / "description.txt")

            if file_txt.exists():
                description = file_txt.read_text().strip()
            else:
                description = "No description available."

            choices.append(make_choice(method, method, description))

        method = prompt_select(
            f"Select a method for the task '{task}':",
            choices=choices,
        )

    if task != BASE_TASK and task not in method_templates:
        raise typer.BadParameter(f"Task '{task}' is not a valid task.")

    if method and method not in method_templates[task]:
        raise typer.BadParameter(f"Method '{method}' is not a valid method for task '{task}'.")

    create_workspace(Path.cwd() / name, task, method)
    create_toml(Path.cwd() / name, name, task, method, description)

