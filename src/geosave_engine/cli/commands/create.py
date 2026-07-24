from __future__ import annotations

from pathlib import Path

import questionary
import typer

from geosave_engine.cli.errors import AbortedByUserError
from geosave_engine.cli.workspace import Workspace, WorkspaceSpec
from geosave_engine.cli.workspace.templates import get_method_templates


def create(
    directory: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Directory to build the GeoSave workspace in.",
    ),
) -> None:
    """Create one GeoSave workspace."""
    workspace = Workspace(directory, _prompt_workspace_spec())
    workspace.setup_workspace()


def _prompt_workspace_spec() -> WorkspaceSpec:
    method_templates = get_method_templates()
    project_name = _ask_required_text("Enter the project name:", "Project name")
    project_task = _ask_required_choice(
        "Select the main task for the project:",
        list(method_templates),
        "Project task",
    )
    project_method = _ask_required_choice(
        "Select the method for the project:",
        list(method_templates[project_task]),
        "Project method",
    )
    description = questionary.text("Enter a description for the project (optional):").ask()

    return WorkspaceSpec(
        project_name=project_name,
        project_task=project_task,
        project_method=project_method,
        description=description.strip() if description else None,
    )


def _ask_required_text(question: str, field_name: str) -> str:
    answer = questionary.text(question).ask()
    if not answer or not answer.strip():
        raise AbortedByUserError(f"{field_name} is required.")
    return answer.strip()


def _ask_required_choice(question: str, choices: list[str], field_name: str) -> str:
    answer = questionary.select(question, choices=choices).ask()
    if answer is None:
        raise AbortedByUserError(f"{field_name} selection was aborted.")
    return answer.strip()
