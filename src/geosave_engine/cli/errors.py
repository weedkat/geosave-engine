class GeosaveCliError(Exception):
    """Base error raised by GeoSave commands."""

    exit_code: int = 1


class AbortedByUserError(GeosaveCliError):
    """The user cancelled a prompt (Ctrl-C, empty answer)."""


class WorkspaceError(GeosaveCliError):
    """A runtime command could not locate or load a workspace."""
