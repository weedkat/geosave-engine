
from __future__ import annotations

from pathlib import Path

from geosave_engine.cli.errors import AbortedByUserError, BuildError
from geosave_engine.cli.io import Console, Prompter
from geosave_engine.cli.paths import plugins_root
from geosave_engine.cli.runtime import Workspace
from geosave_engine.cli.runtime.workspace import announce_workspace, load_workspace
from geosave_engine.cli.search import ProjectLayout
from geosave_engine.utils.cli.fs_ops import copy_tree

_PLUGIN_TYPE_ALIASES = {
    "script": "scripts",
    "scripts": "scripts",
    "notebook": "notebook",
}


def add_plugin(
    project_dir: Path,
    *,
    plugin_type: str | None,
    plugin_name: str | None,
    prompter: Prompter,
    console: Console,
) -> None:
    """Facade for the CLI command; creates a manager and executes add flow."""
    manager = PluginManager(project_dir, prompter=prompter, console=console)
    manager.add_plugin(plugin_type=plugin_type, plugin_name=plugin_name)


class PluginManager:
    """Service for adding a plugin to an existing GeoSave workspace."""

    def __init__(
        self,
        project_dir: Path,
        *,
        prompter: Prompter,
        console: Console,
    ) -> None:
        self.project_dir = project_dir
        self.prompter = prompter
        self.console = console

        self.workspace: Workspace = load_workspace(project_dir)
        announce_workspace(self.workspace, console)
        self.layout = ProjectLayout(root=project_dir)

    def add_plugin(self, plugin_type: str | None = None, plugin_name: str | None = None) -> None:
        """Add the specified plugin to the workspace."""
        selected_type = self._resolve_plugin_type(plugin_type)
        selected_name = self._resolve_plugin_name(selected_type, plugin_name)
        source = self._resolve_source(selected_type, selected_name)
        destination = self._resolve_destination(selected_type, selected_name, source)

        self._confirm_overwrite(source, destination)
        copy_tree(source, destination)
        self.console.success(
            f"Added plugin '{selected_name}' ({selected_type}) to '{destination}'."
        )

    def _resolve_plugin_type(self, plugin_type: str | None) -> str:
        available_types = _discover_plugin_types()
        if not available_types:
            raise BuildError("No plugin template types found under templates/plugins.")

        if plugin_type:
            normalized = _normalize_plugin_type(plugin_type)
            if normalized not in available_types:
                choices = ", ".join(available_types)
                raise BuildError(
                    f"Unknown plugin type '{plugin_type}'. Available types: {choices}."
                )
            return normalized

        selected = self.prompter.select_mapping(
            "Select plugin type:",
            [(name, name) for name in available_types],
        )
        if not selected:
            raise AbortedByUserError("Plugin type selection cancelled.")
        return selected

    def _resolve_plugin_name(self, plugin_type: str, plugin_name: str | None) -> str:
        available_plugins = _discover_plugin_names(plugin_type)
        if not available_plugins:
            raise BuildError(f"No plugins found for type '{plugin_type}'.")

        if plugin_name:
            normalized = _normalize_plugin_name(plugin_type, plugin_name)
            if normalized not in available_plugins:
                choices = ", ".join(available_plugins)
                raise BuildError(
                    f"Unknown plugin '{plugin_name}' for type '{plugin_type}'. "
                    f"Available options: {choices}."
                )
            return normalized

        selected = self.prompter.select_mapping(
            f"Select {plugin_type} plugin:",
            [(name, name) for name in available_plugins],
        )
        if not selected:
            raise AbortedByUserError("Plugin selection cancelled.")
        return selected

    def _resolve_source(self, plugin_type: str, plugin_name: str) -> Path:
        root = plugins_root() / plugin_type
        if plugin_type == "scripts":
            return root / plugin_name

        for candidate in root.glob("*.ipynb"):
            if candidate.stem == plugin_name:
                return candidate

        raise BuildError(f"Could not resolve source for plugin '{plugin_name}'.")

    def _resolve_destination(self, plugin_type: str, plugin_name: str, source: Path) -> Path:
        if plugin_type == "scripts":
            return self.layout.scripts_dir / plugin_name

        return self.layout.root / "notebooks" / source.name

    def _confirm_overwrite(self, source: Path, destination: Path) -> None:
        collisions = _collect_collisions(source, destination)
        if not collisions:
            return

        count = len(collisions)
        preview = ", ".join(path.as_posix() for path in collisions[:3])
        suffix = "" if count <= 3 else f", and {count - 3} more"
        should_overwrite = self.prompter.confirm(
            (
                f"{count} existing file(s) will be overwritten at '{destination}'. "
                f"Examples: {preview}{suffix}. Continue?"
            ),
            default=False,
        )
        if not should_overwrite:
            raise AbortedByUserError("Add plugin cancelled — existing files would be overwritten.")


def _discover_plugin_types() -> list[str]:
    root = plugins_root()
    if not root.is_dir():
        raise BuildError(f"Plugin template root does not exist: '{root}'.")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _discover_plugin_names(plugin_type: str) -> list[str]:
    root = plugins_root() / plugin_type
    if not root.is_dir():
        raise BuildError(f"Plugin type directory does not exist: '{root}'.")

    if plugin_type == "scripts":
        return sorted(path.name for path in root.iterdir() if path.is_dir())

    return sorted(path.stem for path in root.glob("*.ipynb") if path.is_file())


def _normalize_plugin_type(plugin_type: str) -> str:
    normalized = plugin_type.strip().lower()
    return _PLUGIN_TYPE_ALIASES.get(normalized, normalized)


def _normalize_plugin_name(plugin_type: str, plugin_name: str) -> str:
    normalized = plugin_name.strip()
    if plugin_type == "notebook" and normalized.endswith(".ipynb"):
        return Path(normalized).stem
    return normalized


def _collect_collisions(source: Path, destination: Path) -> list[Path]:
    if source.is_file():
        return [destination] if destination.is_file() else []

    if source.is_dir() and destination.is_file():
        return [destination]

    if not source.is_dir():
        return []

    collisions: list[Path] = []
    for candidate in source.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(source)
        target = destination / relative
        if target.is_file():
            collisions.append(target)
    return collisions