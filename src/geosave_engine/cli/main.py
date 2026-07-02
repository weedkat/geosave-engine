import typer
from pathlib import Path
from typing import Optional

from geosave_engine.cli.scaffhold import create_scaffold
from geosave_engine.cli.workspace import Workspace
from geosave_engine.cli.errors import WorkspaceError
from geosave_engine.cli.infra import infra_app

from geosave_engine.cli.prompts import (
    prompt_for_artifact,
)

CURRENT_DIR = Path.cwd()
app = typer.Typer(help="Geosave Engine CLI")
app.add_typer(infra_app, name="infra")


@app.command()
def create(
    dir: str = typer.Option(
        CURRENT_DIR,
        '-d',
        '--dir',
        help="Directory to build the Geosave workspace in")
):
    spec = create_scaffold()
    workspace = Workspace(dir, spec)
    workspace.setup_workspace()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def fit(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(
        None, help="Workspace directory"),
    config: Optional[str] = typer.Option(
        None, '-c', '--config', help="Path to config file"),
):
    work_dir = _get_work_dir(project_dir)
    workspace = Workspace.load_workspace(work_dir)
    args = [*(['-c', config] if config else []), *ctx.args]
    workspace.run_lightning('fit', args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(
        None, help="Workspace directory"),
    config: Optional[str] = typer.Option(
        None, '-c', '--config', help="Path to config file"),
):
    work_dir = _get_work_dir(project_dir)
    workspace = Workspace.load_workspace(work_dir)
    args = [*(['-c', config] if config else []), *ctx.args]
    workspace.run_lightning('test', args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def predict(
    ctx: typer.Context,
    project_dir: Optional[str] = typer.Argument(
        None, help="Workspace directory"),
    artifact: Optional[str] = typer.Option(
        None, '-a', '--artifact', help="Artifact directory (e.g., model_name/version_0)"),
    config: Optional[str] = typer.Option(
        None, '-c', '--config', help="Path to config file"),
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
