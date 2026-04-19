from __future__ import annotations

from typing import Protocol, runtime_checkable

import typer


@runtime_checkable
class Console(Protocol):
    """Output seam so services can be unit-tested with fakes."""

    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def success(self, message: str) -> None: ...


class TyperConsole:
    """Default Console that writes colored output via `typer.secho`."""

    def info(self, message: str) -> None:
        typer.secho(message, fg=typer.colors.CYAN)

    def warn(self, message: str) -> None:
        typer.secho(message, fg=typer.colors.YELLOW)

    def error(self, message: str) -> None:
        typer.secho(message, fg=typer.colors.RED, err=True)

    def success(self, message: str) -> None:
        typer.secho(message, fg=typer.colors.GREEN)
