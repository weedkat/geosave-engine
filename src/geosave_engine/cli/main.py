from __future__ import annotations

from pathlib import Path

import typer

from geosave_engine.cli.errors import BuildError, GeosaveCliError
from geosave_engine.cli.io import QuestionaryPrompter, TyperConsole
from geosave_engine.cli.paths import templates_root
from geosave_engine.cli.generate import collect_build_request, generate_project
from geosave_engine.cli.plugin.plugin import add_plugin
from geosave_engine.cli.runtime import ProjectScriptRunner


app = typer.Typer(help="GeoSave Engine CLI")
CURRENT_DIR = Path.cwd()
_ADD_PLUGIN_TYPE_TOKENS = {"script", "scripts", "notebook"}


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
    and a training method. It copies the necessary templates and generates a ready-to-use
    workspace with a tracking `geosave.toml` file.
    """

    template_dir = templates_root()
    prompter = QuestionaryPrompter()
    console = TyperConsole()

    def _run() -> None:
        request = collect_build_request(
            name,
            template_dir=template_dir,
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


@app.command()
def add(
    project_dir: Path = typer.Argument(
        CURRENT_DIR,
        help="Path to the GeoSave project directory (containing geosave.toml)",
    ),
    plugin_type_arg: str | None = typer.Argument(
        None,
        help="Plugin type to add (e.g., scripts or notebook)",
    ),
    plugin_name_arg: str | None = typer.Argument(
        None,
        help="Plugin name in the selected plugin type",
    ),
    plugin_type: str | None = typer.Option(
        None,
        "--plugin-type",
        "--plugin",
        "-t",
        help="Plugin type to add (supports 'script' alias for 'scripts')",
    ),
    plugin_name: str | None = typer.Option(
        None,
        "--plugin-name",
        "--name",
        "-n",
        help="Plugin name to add",
    ),
):
    """
    Add a new plugin to an existing GeoSave project.

    Usage:
    - geosave add <project_dir> <plugin_type> <plugin_name>
    - geosave add <project_dir> --plugin-type <plugin_type> --plugin-name <plugin_name>
    - geosave add <plugin_type> <plugin_name>  (when already in workspace)
    - geosave add <project_dir>
    """
    prompter = QuestionaryPrompter()
    console = TyperConsole()

    def _run() -> None:
        resolved_project_dir, normalized_type_arg, normalized_name_arg = _normalize_add_positionals(
            project_dir,
            plugin_type_arg,
            plugin_name_arg,
        )
        resolved_type = _resolve_add_value("plugin type", normalized_type_arg, plugin_type)
        resolved_name = _resolve_add_value("plugin name", normalized_name_arg, plugin_name)
        add_plugin(
            resolved_project_dir,
            plugin_type=resolved_type,
            plugin_name=resolved_name,
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
    return ProjectScriptRunner(
        project_dir=project_dir,
        current_dir=CURRENT_DIR,
        prompter=QuestionaryPrompter(),
        console=TyperConsole(),
    )


def _resolve_add_value(
    label: str,
    positional: str | None,
    option: str | None,
) -> str | None:
    if positional and option and positional != option:
        raise BuildError(
            f"Conflicting {label}: positional '{positional}' does not match option '{option}'."
        )
    return option or positional


def _normalize_add_positionals(
    project_dir: Path,
    plugin_type_arg: str | None,
    plugin_name_arg: str | None,
) -> tuple[Path, str | None, str | None]:
    """Support `geosave add <plugin_type> <plugin_name>` when run in a workspace."""
    if plugin_name_arg is not None:
        return project_dir, plugin_type_arg, plugin_name_arg

    token = project_dir.as_posix().strip().lower()
    looks_like_plugin_type = token in _ADD_PLUGIN_TYPE_TOKENS
    has_workspace_manifest = (project_dir / "geosave.toml").is_file()

    if looks_like_plugin_type and not has_workspace_manifest:
        return CURRENT_DIR, token, plugin_type_arg

    return project_dir, plugin_type_arg, plugin_name_arg


if __name__ == "__main__":
    app()
