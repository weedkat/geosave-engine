from __future__ import annotations


class GeosaveCliError(Exception):
    """Base class for CLI service errors caught and surfaced in main.py."""

    exit_code: int = 1


class AbortedByUserError(GeosaveCliError):
    """The user cancelled a prompt (Ctrl-C, empty answer)."""


class BuildError(GeosaveCliError):
    """Scaffolding or project generation failed."""


class WorkspaceError(GeosaveCliError):
    """A runtime command could not locate or load a workspace."""


class ScriptError(GeosaveCliError):
    """A subprocess invocation failed; `exit_code` is the process return code."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class DocsError(GeosaveCliError):
    """An unknown docs section or lookup failure."""
