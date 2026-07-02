import typer
from pathlib import Path
from typing import Optional

from geosave_engine.cli.scaffhold import create_scaffold
from geosave_engine.cli.workspace import Workspace
from geosave_engine.cli.errors import WorkspaceError
from geosave_engine.cli.paths import get_plugin_templates

from geosave_engine.cli.prompts import (
    prompt_for_plugin,
    prompt_for_runnable,
    prompt_for_artifact,
)

CURRENT_DIR = Path.cwd()
app = typer.Typer(help="Geosave Engine CLI")


@app.command()
def create(
    dir: str = typer.Option(
        CURRENT_DIR,
        '-d',
        '--dir',
        help="Directory to build the Geosave Engine in")
):
    spec = create_scaffold()
    workspace = Workspace(dir, spec)
    workspace.setup_workspace()


@app.command()
def add(
    project_dir: Optional[str] = typer.Argument(None, help="Workspace directory"),
    plugin_path: Optional[str] = typer.Argument(
        None, help="Namespaced plugin path, e.g. 'scripts/dynamicworld' or 'notebooks/tutorial'"),
    flat: bool = typer.Option(
        False,
        '--flat', '-f',
        help="Copy plugin directly into project root instead of its namespace subdirectory"),
):
    work_dir = _get_work_dir(project_dir)
    workspace = Workspace.load_workspace(work_dir)
    plugin_templates = get_plugin_templates()

    if not plugin_templates:
        raise WorkspaceError("No plugins available.")

    if plugin_path is None:
        plugin_path = prompt_for_plugin(plugin_templates)

    if plugin_path not in plugin_templates:
        raise WorkspaceError(
            f"Plugin {plugin_path!r} not found. Available: {', '.join(sorted(plugin_templates))}"
        )

    workspace.add_plugin(plugin_templates[plugin_path], flat=flat)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(None, help="Workspace directory"),
    name: Optional[str] = typer.Option(
        None, '-s', '--script', help="Script or notebook name to run"),
):
    work_dir = _get_work_dir(project_dir)
    workspace = Workspace.load_workspace(work_dir)

    scripts = {f"scripts/{k}": v for k, v in workspace.scripts.items()}
    notebooks = {f"notebooks/{k}": v for k, v in workspace.notebooks.items()}
    runnables = {**scripts, **notebooks}

    if name is None:
        if not runnables:
            raise WorkspaceError(
                f"No scripts or notebooks found in: {workspace.scripts_dir}, {workspace.notebooks_dir}"
            )
        name = prompt_for_runnable(sorted(runnables.keys()))

    # Normalize: strip leading ./ and optional scripts/ or notebooks/ prefix for lookup
    key = name.lstrip("./")
    if key not in runnables:
        # try adding namespace prefix if user passed bare name
        for prefix in ("scripts/", "notebooks/"):
            candidate = f"{prefix}{key}"
            if candidate in runnables:
                key = candidate
                break
        else:
            raise WorkspaceError(
                f"{name!r} not found. Available: {', '.join(sorted(runnables))}"
            )

    path = runnables[key]
    if path.suffix == ".ipynb":
        workspace.run_notebook(path, ctx.args)
    else:
        workspace.run_script(path, ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def fit(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(
        None, help="Directory of the Geosave Engine project to fit"),
    config: Optional[str] = typer.Option(
        None, '-c', '--config', help="Path to a configuration file for fitting the Geosave Engine")
):
    work_dir = _get_work_dir(project_dir)
    workspace = Workspace.load_workspace(work_dir)
    args = [*(['-c', config] if config else []), *ctx.args]
    workspace.run_lightning('fit', args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(
        CURRENT_DIR, help="Directory of the Geosave Engine project to test"),
    config: Optional[str] = typer.Option(
        None, '-c', '--config', help="Path to a configuration file for testing the Geosave Engine")
):
    work_dir = _get_work_dir(project_dir)
    workspace = Workspace.load_workspace(work_dir)
    args = [*(['-c', config] if config else []), *ctx.args]
    workspace.run_lightning('test', args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def predict(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(
        CURRENT_DIR, help="Directory of the Geosave Engine project to predict"),
    artifact: Optional[str] = typer.Option(
        None, '-a', '--artifact', help="Artifact directory (e.g., model_name/version_0)"),
    config: Optional[str] = typer.Option(
        None, '-c', '--config', help="Path to a configuration file for predicting with the Geosave Engine")
):
    work_dir = _get_work_dir(project_dir)
    workspace = Workspace.load_workspace(work_dir)

    if artifact is None:
        if not workspace.artifacts:
            raise WorkspaceError(f"No artifacts found in: {workspace.artifacts_dir}")
        artifact = prompt_for_artifact(list(workspace.artifacts.keys()))

    if artifact in workspace.artifacts:
        config_path = workspace.artifacts[artifact]
    elif config:
        config_path = Path(config)
    else:
        raise WorkspaceError(
            f"Artifact not found: {artifact}. Available: {', '.join(sorted(workspace.artifacts.keys()))}"
        )

    args = ['-c', str(config_path), *ctx.args]
    workspace.run_lightning('predict', args)


def _get_work_dir(dir: str | None) -> Path:
    if dir is None:
        return CURRENT_DIR
    return CURRENT_DIR / dir
