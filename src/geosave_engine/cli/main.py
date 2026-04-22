from __future__ import annotations

from pathlib import Path

import typer

from geosave_engine.cli.errors import GeosaveCliError
from geosave_engine.cli.io import QuestionaryPrompter, TyperConsole
from geosave_engine.cli.paths import models_root, templates_root
from geosave_engine.cli.generate import collect_build_request, generate_project


app = typer.Typer(help="GeoSave Engine CLI")
CURRENT_DIR = Path.cwd()


def _handle(func):
    """Catch `GeosaveCliError` raised by services and exit with its code."""
    console = TyperConsole()
    try:
        func()
    except GeosaveCliError as error:
        console.error(str(error))
        raise typer.Exit(error.exit_code) from error


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

    template_dir = templates_root()
    models_package_path = models_root()
    prompter = QuestionaryPrompter()
    console = TyperConsole()

    def _run() -> None:
        request = collect_build_request(
            name,
            template_dir=template_dir,
            models_package_path=models_package_path,
            prompter=prompter,
            console=console,
        )
        generate_project(
            request,
            output_dir=Path(dir),
            template_dir=template_dir,
            prompter=prompter,
            console=console,
        )

    _handle(_run)


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
    _handle(lambda: _make_runner(project_dir).fit(config=config, extra_args=ctx.args))


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
    _handle(
        lambda: _make_runner(project_dir).test(artifacts=artifacts, extra_args=ctx.args)
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
    _handle(
        lambda: _make_runner(project_dir).predict(
            artifacts=artifacts, extra_args=ctx.args
        )
    )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    ctx: typer.Context,
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
    script_name: str | None = typer.Option(
        None,
        "--script",
        "-s",
        help="Script name in project scripts/ directory (with or without .py)",
    ),
):
    """
    Execute a custom script from the project's scripts directory.

    Usage:
    - geosave run
    - geosave run <project_dir>
    - geosave run <project_dir> --script <script_name>
    - geosave run <project_dir> <script_name> [<extra_args>...]

    The script name can be given via --script or as the first positional extra
    argument (with or without the .py suffix). If omitted, you will be prompted
    to select one from scripts/. Any extra flags are always forwarded to the script.
    """
    _handle(
        lambda: _make_runner(project_dir).run(
            script_name=script_name, extra_args=ctx.args
        )
    )


@app.command()
def docs(
    section: str | None = typer.Argument(
        None,
        help="Docs section: lightningmodule | datamodule | trainer | templates | model | loss | optimizer",
    ),
    arg1: str | None = typer.Argument(
        None,
        help="Optional chained arg (e.g., task for model, name for loss/optimizer)",
    ),
    arg2: str | None = typer.Argument(
        None,
        help="Optional chained arg (e.g., method for model)",
    ),
    arg3: str | None = typer.Argument(
        None,
        help="Optional chained arg (e.g., model name)",
    ),
):
    """
    Quick CLI reference for GeoSave components.

    Examples:
    - geosave docs
    - geosave docs lightningmodule
    - geosave docs datamodule
    - geosave docs model semantic_segmentation supervised
    - geosave docs optimizer AdamW
    """
    from geosave_engine.cli.docs import show_docs

    prompter = QuestionaryPrompter()
    _handle(
        lambda: show_docs(
            section=section,
            arg1=arg1,
            arg2=arg2,
            arg3=arg3,
            prompter=prompter,
        )
    )


def _make_runner(project_dir: Path):
    from geosave_engine.cli.runtime import ProjectScriptRunner

    return ProjectScriptRunner(
        project_dir=project_dir,
        current_dir=CURRENT_DIR,
        prompter=QuestionaryPrompter(),
        console=TyperConsole(),
    )


if __name__ == "__main__":
    app()
