from __future__ import annotations
import typer
from pathlib import Path

from geosave_engine.cli.utils.search import find_script
from geosave_engine.cli.utils.execute import execute_project_script
from geosave_engine.cli.utils.cli_helpers import (
    validate_workspace_with_feedback,
    get_config_args,
    get_artifacts_args,
)
from geosave_engine.cli.build import build_project


app = typer.Typer(help="GeoSave Engine CLI")
CURRENT_DIR = Path.cwd()
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"

@app.command()
def build(
    name: str | None = typer.Argument(
        None, help="The name of the new GeoSave project to create"
    ),
    dir: str = typer.Option(
        str(CURRENT_DIR),
        "--dir",
        "-d",
        help="The parent directory where the project will be created",
    ),
):
    """
    Build a new GeoSave project workspace.

    This command interactively scaffolds a new project by asking you to select an AI task,
    a training method, and the specific models you want to use. It copies the necessary
    templates and generates a ready-to-use workspace with a tracking `geosave.toml` file.
    """
    build_project(name, dir, TEMPLATE_DIR)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def fit(
    ctx: typer.Context,
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the configuration file (e.g., config.yml) used for training",
    ),
):
    """
    Train models in a GeoSave workspace.

    Executes the training pipeline inside the provided workspace. If no `--config` is provided,
    the command will scan the workspace folder for `.yaml` or `.yml` configuration files and
    prompt you to select one interactively.
    """
    workspace_config = validate_workspace_with_feedback(project_dir)

    train_script = find_script(project_dir, "main.py")

    run_args = ['fit']
    run_args.extend(get_config_args(project_dir, config))
    run_args.extend(ctx.args)

    execute_project_script(
        project_name=workspace_config.get("project_name", "Unknown"),
        task_name=workspace_config.get("task", ""),
        script_path=train_script,
        project_dir=project_dir,
        current_dir=CURRENT_DIR,
        run_args=run_args,
        operation="training",
    )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def test(
    ctx: typer.Context,
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
    artifacts: str | None = typer.Option(
        None,
        "--artifacts",
        "-a",
        help="Path to a specific model artifacts directory to evaluate against",
    ),
):
    """
    Test models evaluated under a GeoSave pipeline.

    This executes the testing logic using your generated `train.py test` script.
    It requires selecting an artifact directory inside `artifacts/` that contains
    saved model checkpoints and configuration files. If `--artifacts` isn't found,
    the CLI prompts you interactively.
    """
    workspace_config = validate_workspace_with_feedback(project_dir)

    test_script = find_script(project_dir, "main.py")

    run_args = ['test']
    run_args.extend(get_artifacts_args(project_dir, artifacts))
    run_args.extend(ctx.args)

    execute_project_script(
        project_name=workspace_config.get("project_name", "Unknown"),
        task_name=workspace_config.get("task", ""),
        script_path=test_script,
        project_dir=project_dir,
        current_dir=CURRENT_DIR,
        run_args=run_args,
        operation="test",
    )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def predict(
    ctx: typer.Context,
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
    artifacts: str | None = typer.Option(
        None,
        "--artifacts",
        "-a",
        help="Path to the trained model artifacts directory used for predictions",
    ),
):
    """
    Run predictions inside a GeoSave workspace.

    This command invokes `inference.py` inside the project folder. It requires specifying
    which artifact folder (containing your trained weights and config metadata) to use
    to properly reconstruct the model before testing your data.
    """
    workspace_config = validate_workspace_with_feedback(project_dir)

    infer_script = find_script(project_dir, "main.py")

    run_args = ['predict']
    run_args.extend(get_artifacts_args(project_dir, artifacts))
    run_args.extend(ctx.args)

    execute_project_script(
        project_name=workspace_config.get("project_name", "Unknown"),
        task_name=workspace_config.get("task", ""),
        script_path=infer_script,
        project_dir=project_dir,
        current_dir=CURRENT_DIR,
        run_args=run_args,
        operation="inference",
    )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def ingest(
    ctx: typer.Context,
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
):
    """
    Run data ingestion in a GeoSave workspace.

    Executes the ingestion pipeline. You can pass arbitrary flags to the script.
    """
    workspace_config = validate_workspace_with_feedback(project_dir)

    ingest_script = find_script(project_dir, "ingest.py")

    execute_project_script(
        project_name=workspace_config.get("project_name", "Unknown"),
        task_name=workspace_config.get("task", ""),
        script_path=ingest_script,
        project_dir=project_dir,
        current_dir=CURRENT_DIR,
        run_args=ctx.args,
        operation="ingestion",
    )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def preprocess(
    ctx: typer.Context,
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
):
    """
    Run data preprocessing in a GeoSave workspace.

    Executes the preprocessing pipeline. You can pass arbitrary flags to the script.
    """
    workspace_config = validate_workspace_with_feedback(project_dir)

    preprocess_script = find_script(project_dir, "preprocess.py")

    execute_project_script(
        project_name=workspace_config.get("project_name", "Unknown"),
        task_name=workspace_config.get("task", ""),
        script_path=preprocess_script,
        project_dir=project_dir,
        current_dir=CURRENT_DIR,
        run_args=ctx.args,
        operation="preprocessing",
    )


@app.command()
def upload(
    ctx: typer.Context,
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
):
    """
    Upload a model to Hugging Face Hub.

    This command looks for an `upload.py` script in the current directory and executes it.
    You can pass arbitrary flags to the upload script.
    """
    pass
    # workspace_config = _validate_workspace(CURRENT_DIR)

    # upload_model()


@app.command()
def docs():
    """
    Open the GeoSave Engine documentation in your web browser.
    """
    # execute_script()


if __name__ == "__main__":
    app()
