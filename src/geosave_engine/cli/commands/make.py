import typer
from typing import Annotated, Optional
from pathlib import Path

from geosave_engine.utils.file_ops import safe_copy

from ..core.templates import get_boilerplate, boilerplate_dir
from ..core.prompts import prompt_select
from ..core.workspace import Workspace


def make(
    boilerplate: Annotated[
        Optional[str],
        typer.Argument(
            help="The name of the boilerplate to scaffold.",
        ),
    ] = None,
    filename: Annotated[
        Optional[str],
        typer.Argument(
            help="The name of the new component.",
        ),
    ] = None,
):
    workspace = Workspace(Path.cwd())
    boilerplates = get_boilerplate()

    if boilerplate is None:
        boilerplate = prompt_select("Select a boilerplate:", list(boilerplates.keys()))

    if filename is None:
        filename = prompt_select(
            f"Select a file to scaffold for the boilerplate '{boilerplate}':",
            boilerplates[boilerplate],
        )

    if boilerplate not in boilerplates:
        raise typer.BadParameter(f"Boilerplate '{boilerplate}' is not a valid boilerplate.")

    if filename not in boilerplates[boilerplate]:
        raise typer.BadParameter(f"File '{filename}' is not a valid file for boilerplate '{boilerplate}'.")

    src_path = boilerplate_dir() / boilerplate / filename
    dest_path = workspace.root / boilerplate / filename

    safe_copy(src_path, dest_path)