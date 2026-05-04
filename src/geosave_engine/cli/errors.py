class GeosaveCliError(Exception):
    """Base class for CLI service errors caught and surfaced in main.py."""

    exit_code: int = 1

class AbortedByUserError(GeosaveCliError):
    """The user cancelled a prompt (Ctrl-C, empty answer)."""

class WorkspaceError(GeosaveCliError):
    """A runtime command could not locate or load a workspace."""
