from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
import questionary as qu

from ..core.templates import TASK_DIR, get_tasks
from ..core.workspace import create_workspace
from ..core.toml import create_toml
from ..core.prompts import prompt_required_text, prompt_optional_text, prompt_select

NO_TASK = "blank"


def _template_description(directory: Path) -> str:
    """Read a template's `description.txt`, or a placeholder when it has none.

    Args:
        directory: Task or method template directory.

    Returns:
        The description text, stripped.
    """
    description_file = directory / "description.txt"
    if description_file.exists():
        return description_file.read_text().strip()
    return "No description available."


def create(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Project name to create the workspace in."),
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
    methods_by_task = get_tasks()

    if task is not None and task != NO_TASK and task not in methods_by_task:
        raise typer.BadParameter(f"Task '{task}' is not a valid task.")

    if name is None:
        name = prompt_required_text("Enter a name for the workspace:")

    if description is None:
        description = prompt_optional_text(
            "Enter a description for the workspace (optional):"
        )

    if task is None:
        choices = [
            qu.Choice(title=NO_TASK, value=NO_TASK, description="No task selected.")
        ]
        for task_name in methods_by_task:
            choices.append(
                qu.Choice(
                    title=task_name,
                    value=task_name,
                    description=_template_description(TASK_DIR / task_name),
                )
            )
        task = prompt_select("Select a task for the workspace:", choices=choices)

    if task == NO_TASK:
        if method is not None:
            raise typer.BadParameter(f"Task '{NO_TASK}' takes no method.")
    else:
        if method is None:
            choices = []
            for method_name in methods_by_task[task]:
                choices.append(
                    qu.Choice(
                        title=method_name,
                        value=method_name,
                        description=_template_description(
                            TASK_DIR / task / method_name
                        ),
                    )
                )
            method = prompt_select(
                f"Select a method for the task '{task}':",
                choices=choices,
            )
        elif method not in methods_by_task[task]:
            raise typer.BadParameter(
                f"Method '{method}' is not a valid method for task '{task}'."
            )

    root = Path.cwd() / name
    chosen_task = None if task == NO_TASK else task
    create_workspace(root, chosen_task, method)
    create_toml(root, name, chosen_task, method, description)
