import typer
from typing import Annotated, Optional
from pathlib import Path

from geosave_engine.utils.file_ops import safe_copy

from ..core.templates import BOILERPLATE_DIR, get_boilerplate
from ..core.prompts import prompt_select


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
            help="File to copy out of the boilerplate.",
        ),
    ] = None,
) -> None:
    """Copy one boilerplate file into the current workspace."""
    root = Path.cwd()
    if not (root / "geosave.toml").exists():
        raise typer.BadParameter(f"geosave.toml not found in {root}")

    boilerplates = get_boilerplate()

    if boilerplate is None:
        boilerplate = prompt_select("Select a boilerplate:", list(boilerplates))
    elif boilerplate not in boilerplates:
        raise typer.BadParameter(
            f"Boilerplate '{boilerplate}' is not a valid boilerplate."
        )

    if filename is None:
        filename = prompt_select(
            f"Select a file to scaffold for the boilerplate '{boilerplate}':",
            boilerplates[boilerplate],
        )
    elif filename not in boilerplates[boilerplate]:
        raise typer.BadParameter(
            f"File '{filename}' is not a valid file for boilerplate '{boilerplate}'."
        )

    safe_copy(BOILERPLATE_DIR / boilerplate / filename, root / boilerplate / filename)
