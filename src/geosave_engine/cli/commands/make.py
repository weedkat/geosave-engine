import typer
from typing import Annotated, Optional
from pathlib import Path

from geosave_engine.cli.core.copy import safe_copy

from ..core.templates import SCAFFOLDS_DIR, get_scaffolds
from ..core.prompts import prompt_select


def make(
    scaffold: Annotated[
        Optional[str],
        typer.Argument(
            help="Destination category of the scaffold, such as scripts.",
        ),
    ] = None,
    filename: Annotated[
        Optional[str],
        typer.Argument(
            help="File to copy out of the scaffold.",
        ),
    ] = None,
) -> None:
    """Copy one scaffold file into the current workspace."""
    root = Path.cwd()
    if not (root / "geosave.toml").exists():
        raise typer.BadParameter(f"geosave.toml not found in {root}")

    scaffolds = get_scaffolds()

    if scaffold is None:
        scaffold = prompt_select("Select a scaffold:", list(scaffolds))
    elif scaffold not in scaffolds:
        raise typer.BadParameter(f"Scaffold '{scaffold}' is not a valid scaffold.")

    if filename is None:
        filename = prompt_select(
            f"Select a file for '{scaffold}':",
            scaffolds[scaffold],
        )
    elif filename not in scaffolds[scaffold]:
        raise typer.BadParameter(
            f"File '{filename}' is not a valid file for scaffold '{scaffold}'."
        )

    safe_copy(SCAFFOLDS_DIR / scaffold / filename, root / scaffold / filename)
