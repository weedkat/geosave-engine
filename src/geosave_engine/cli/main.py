import typer
from pathlib import Path
from typing import Optional

from geosave_engine.cli.scaffhold import create_scaffold
from geosave_engine.cli.workspace import Workspace
from geosave_engine.cli.errors import WorkspaceError
from geosave_engine.cli.paths import (
    plugins,
    get_plugin_templates,
)

from geosave_engine.cli.prompts import (
    prompt_for_plugin_type,
    prompt_for_plugin_name,
    prompt_for_script_name,
    prompt_for_artifact,
)

CURRENT_DIR = Path.cwd()
app = typer.Typer(help="Geosave Engine CLI")

@app.command()
def build(
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
    project_dir: Optional[str] = typer.Argument(None, help="Directory of the Geosave Engine project to add a component to"),
    plugin_type: Optional[plugins] = typer.Argument(None, help="Type of component to add (e.g., 'script', 'notebook')"),
    plugin_name: Optional[str] = typer.Argument(None, help="Name of the component to add"),
    flat: bool = typer.Option(
        False,
        '--flat', '-f',
        help="Whether to add the plugin directly to the project root instead of within a subdirectory")
):  
    work_dir = _get_work_dir(project_dir)
    
    workspace = Workspace.load_workspace(work_dir)

    if plugin_type is None:
        plugin_type = prompt_for_plugin_type()

    task = workspace.spec.project_task
    plugin_templates = get_plugin_templates(plugin_type)[task]
    
    if plugin_name is None:
        plugin_name = prompt_for_plugin_name(plugin_type, task) 
    plugin_path = plugin_templates[plugin_name]

    workspace.add_plugin(plugin_path, flat=flat)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(
        None, help="Directory of the Geosave Engine project to run"),
    script_name: Optional[str] = typer.Option(
        None, '-s', '--script', help="Specific script to run within the Geosave Engine project")
):
    work_dir = _get_work_dir(project_dir)

    workspace = Workspace.load_workspace(work_dir)

    if script_name is None:
        if not workspace.scripts:
            raise WorkspaceError(f"No runnable scripts found in: {workspace.scripts_dir}")
        script_name = prompt_for_script_name(list(workspace.scripts.keys()))

    # Normalize input and resolve solely via discovered mapping.
    key = str(script_name).lstrip("./")
    if key.startswith("scripts/"):
        key = key.split("/", 1)[1]
    candidates = (key,) if key.endswith(".py") else (key, f"{key}.py")
    for c in candidates:
        if c in workspace.scripts:
            script_path = workspace.scripts[c]
            break
    else:
        raise WorkspaceError(f"Script not found. Available: {', '.join(sorted(workspace.scripts.keys()))}")

    workspace.run_script(script_path, ctx.args)


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

    # Resolve artifact
    if artifact is None:
        if not workspace.artifacts:
            raise WorkspaceError(f"No artifacts found in: {workspace.artifacts_dir}")
        artifact = prompt_for_artifact(list(workspace.artifacts.keys()))

    # Resolve config: if artifact provided, read config from artifact; else use --config
    if artifact in workspace.artifacts:
        config_path = workspace.artifacts[artifact]
    elif config:
        config_path = Path(config)
    else:
        raise WorkspaceError(f"Artifact not found: {artifact}. Available: {', '.join(sorted(workspace.artifacts.keys()))}")

    args = ['-c', str(config_path), *ctx.args]
    workspace.run_lightning('predict', args)

def _get_work_dir(dir: str | None) -> Path:
    if dir is None:
        return CURRENT_DIR
    else:
        return CURRENT_DIR / dir