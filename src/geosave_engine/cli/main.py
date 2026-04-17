from __future__ import annotations
import typer
from pathlib import Path

from geosave_engine.cli.services.build.project_generation_service import (
    ProjectGenerationService,
)
from geosave_engine.cli.services.build.scaffold_service import BuildScaffoldService
from geosave_engine.cli.services.runtime.project_script_runner import ProjectScriptRunner
from geosave_engine.cli.metadata import models_root, templates_root
from geosave_engine.cli.services.docs.docs_generator import show_docs


app = typer.Typer(help="GeoSave Engine CLI")
CURRENT_DIR = Path.cwd()
TEMPLATE_DIR = templates_root()
MODELS_PACKAGE_PATH = models_root()

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
    scaffold_service = BuildScaffoldService(
        template_dir=TEMPLATE_DIR,
        models_package_path=MODELS_PACKAGE_PATH,
    )
    request = scaffold_service.collect_request(name)
    created = ProjectGenerationService.generate_project(
        request=request,
        output_dir=dir,
        template_dir=TEMPLATE_DIR,
    )
    if not created:
        raise typer.Exit(1)


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
    runner = ProjectScriptRunner(project_dir=project_dir, current_dir=CURRENT_DIR)
    runner.fit(config=config, extra_args=ctx.args)


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
    runner = ProjectScriptRunner(project_dir=project_dir, current_dir=CURRENT_DIR)
    runner.test(artifacts=artifacts, extra_args=ctx.args)


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
    runner = ProjectScriptRunner(project_dir=project_dir, current_dir=CURRENT_DIR)
    runner.predict(artifacts=artifacts, extra_args=ctx.args)


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

    If script is not specified, you'll be prompted to select one from scripts/.
    Any extra flags are passed through to the script. If no flags are passed,
    you'll be prompted to enter them interactively.
    """
    runner = ProjectScriptRunner(project_dir=project_dir, current_dir=CURRENT_DIR)
    runner.run(script_name=script_name, extra_args=ctx.args)


@app.command()
def docs(
    section: str | None = typer.Argument(
        None,
        help="Docs section: lightningmodule | datamodule | model | loss | optimizer",
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
    show_docs(section=section, arg1=arg1, arg2=arg2, arg3=arg3)


if __name__ == "__main__":
    app()
