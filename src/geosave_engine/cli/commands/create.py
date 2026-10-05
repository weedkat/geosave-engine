from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
import questionary as qu

from ..core.templates import WORKSPACES_DIR, get_workspaces
from ..core.workspace import create_workspace
from ..core.toml import create_toml
from ..core.prompts import prompt_required_text, prompt_optional_text, prompt_select

BLANK_WORKSPACE = "blank"


def _template_description(directory: Path) -> str:
    """Read a template's `description.txt`, or a placeholder when it has none.

    Args:
        directory: Workspace starter directory.

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
    workspace: Annotated[
        Optional[str],
        typer.Option("-w", "--workspace", help="Workspace starter, or blank."),
    ] = None,
) -> None:
    """Create one editable GeoSave workspace."""
    workspaces = get_workspaces()
    if (
        workspace is not None
        and workspace != BLANK_WORKSPACE
        and workspace not in workspaces
    ):
        raise typer.BadParameter(f"Workspace {workspace!r} is not a valid workspace.")

    if name is None:
        name = prompt_required_text("Enter a name for the workspace:")
    if description is None:
        description = prompt_optional_text(
            "Enter a description for the workspace (optional):"
        )
    if workspace is None:
        choices = [
            qu.Choice(
                title=BLANK_WORKSPACE,
                value=BLANK_WORKSPACE,
                description="Shared startup only.",
            )
        ]
        choices.extend(
            qu.Choice(
                title=starter,
                value=starter,
                description=_template_description(WORKSPACES_DIR / starter),
            )
            for starter in workspaces
        )
        workspace = prompt_select("Select a workspace starter:", choices=choices)

    root = Path.cwd() / name
    starter = None if workspace == BLANK_WORKSPACE else workspace
    create_workspace(root, starter)
    create_toml(root, name, starter, description)
