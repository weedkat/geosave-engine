from __future__ import annotations
import subprocess
import toml
import questionary
import os
import typer
from pathlib import Path
from geosave_engine.utils.folder_parse import get_model_list, tasks, validate_workspace
from geosave_engine.utils.boilerplate import copier, generate_models_file


app = typer.Typer(help="GeoSave Engine CLI")
CURRENT_DIR = Path.cwd()


@app.command()
def build(
    name: str | None = typer.Argument(None, help="The name of the new GeoSave project to create"),
    dir: str = typer.Option(str(CURRENT_DIR), "--dir", "-d", help="The parent directory where the project will be created")
):
    """
    Build a new GeoSave project workspace.
    
    This command interactively scaffolds a new project by asking you to select an AI task, 
    a training method, and the specific models you want to use. It copies the necessary 
    templates and generates a ready-to-use workspace with a tracking `geosave.toml` file.
    """
    print("Building the project...")
    
    if name is None:
        name = questionary.text(
            "Enter the name of the build:",
            validate=lambda text: bool(text.strip()) or "Name cannot be empty"
        ).ask()

    task = questionary.select(
        "Select the AI task:",
        choices=[t for t in tasks]
    ).ask()
    
    methods = tasks[task]
    
    method = questionary.select(
        "Select the methods:",
        choices=[m for m in methods]
    ).ask()
    
    model_choices = get_model_list(task, method)
    
    if not model_choices:
        print("No models found for the selected task and method.")
        return

    models = questionary.checkbox(
        "Select the models:",
        choices=model_choices
    ).ask()

    description = questionary.text(
        "Enter the description of the build:",
        default="A GeoSave Engine project.",
    ).ask()
    
    template_dir = Path(__file__).parent.parent / "templates" / task.replace(" ", "_") / method.replace(" ", "_")
    
    try:
        if copier(str(template_dir), os.path.join(dir, name)):
            os.makedirs(os.path.join(dir, name, "data"), exist_ok=True)
            os.makedirs(os.path.join(dir, name, "artifacts"), exist_ok=True)
        else:
            return

        try:
            generate_models_file(models, os.path.join(dir, name, "src", "model_factory.py"))
        except Exception as e:
            print(f"An error occurred during model file generation: {e}")
            return

    except Exception as e:
        print(f"An error occurred during copying: {e}")
        return

    env_path = Path(__file__).parent.parent / "templates" / ".env"
    copier(str(env_path), os.path.join(dir, ".env"))

    with open(os.path.join(dir, name, "geosave.toml"), "w", encoding="utf-8") as f:
        toml.dump({
            "project_name": name,
            "task": task,
            "method": method,
            "models": models,
            "description": description
        }, f)


def _get_run_args(project_dir: Path, config: str | None) -> list[str]:
    args = []
    if config:
        args.extend(["--config", config])
    else:
        configs_dir = project_dir / "configs"
        config_files = []
        if configs_dir.exists() and configs_dir.is_dir():
            config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
            
        if config_files:
            choices = [questionary.Choice(f.name, value=str(f.resolve())) for f in config_files]
            
            selected = questionary.select(
                "Select the configuration file:",
                choices=choices
            ).ask()
            
            if selected:
                args.extend(["--config", selected])
            else:
                # User pressed Ctrl+C
                raise typer.Exit(1)
        else:
            typer.secho(f"Error: No .yaml or .yml configuration files found in '{project_dir / 'configs'}'.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    return args


def _get_artifacts_args(project_dir: Path, artifacts: str | None) -> list[str]:
    args = []
    if artifacts:
        args.extend(["--model", artifacts])
    else:
        artifacts_path = project_dir / "artifacts"
        if artifacts_path.exists() and artifacts_path.is_dir():
            choices = []
            for d in artifacts_path.iterdir():
                if d.is_dir() and (list(d.glob("*.yaml")) or list(d.glob("*.yml"))):
                    choices.append(questionary.Choice(d.name, value=str(d.resolve())))
            
            if choices:
                selected = questionary.select(
                    "Select the model",
                    choices=choices
                ).ask()
                
                if selected:
                    args.extend(["--model", selected])
                else:
                    # User pressed Ctrl+C
                    raise typer.Exit(1)
            else:
                typer.secho(f"Warning: No valid artifact directories containing config files found in '{artifacts_path}'.", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"Warning: Artifacts directory not found at '{artifacts_path}'.", fg=typer.colors.YELLOW)
    return args

def _build_env(project_dir: Path, env_vars: list[str] | None = None) -> dict[str, str]:
    """Build the environment dictionary for subprocess execution."""
    env = os.environ.copy()
    
    # Auto-load optionally provided .env file in the current working directory
    env_file = CURRENT_DIR / ".env"
    if env_file.exists() and env_file.is_file():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    # Ignore empty lines and comments
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        # Remove whitespace and surrounding quotes
                        env[k.strip()] = v.strip().strip("'\"")
        except Exception as e:
            typer.secho(f"Warning: Failed to load .env file at {env_file}: {e}", fg=typer.colors.YELLOW)

    # Inject default paths that scripts might find helpful
    env["GEOSAVE_PROJECT_DIR"] = str(project_dir.resolve())
    
    # Inject user-provided env flags
    if env_vars:
        for ev in env_vars:
            if "=" in ev:
                k, v = ev.split("=", 1)
                env[k] = v
            else:
                typer.secho(f"Warning: Invalid environment variable format '{ev}'. Expected KEY=VALUE.", fg=typer.colors.YELLOW)
    return env

@app.command()
def train(
    project_dir: Path = typer.Argument(CURRENT_DIR, help="Path to the GeoSave project directory (containing geosave.toml)"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to the configuration file (e.g., config.yml) used for training"),
    env_vars: list[str] | None = typer.Option(None, "--env", "-e", help="Set environment variable(s) for the script (format: KEY=VALUE)")
):
    """
    Train models in a GeoSave workspace.
    
    Executes the training pipeline inside the provided workspace. If no `--config` is provided, 
    the command will scan the workspace folder for `.yaml` or `.yml` configuration files and 
    prompt you to select one interactively.
    """
    workspace_config = validate_workspace(project_dir)
        
    project_name = workspace_config.get("project_name", "Unknown Project")
    if isinstance(project_name, tuple):
        project_name = project_name[0]
        
    typer.secho(f"Found GeoSave project workspace: '{project_name}'", fg=typer.colors.CYAN)
    
    train_script = project_dir / "train.py"

    if not train_script.exists():
        typer.secho(f"Error: train.py not found in {project_dir}", fg=typer.colors.RED, err=True)
        return
        
    run_args = _get_run_args(project_dir, config)
    cmd_env = _build_env(project_dir, env_vars)
    args_str = " ".join(run_args) if run_args else "No extra args"
    typer.secho(f"Starting training for '{project_name}' (Task: {workspace_config.get('task')})\nExecuting: `python train.py {args_str}`", fg=typer.colors.GREEN)
    
    try:
        # Run subprocess in the valid project_dir workspace, appending arbitrary passed arguments
        subprocess.run(["python", str(train_script.resolve())] + run_args, check=True, cwd=project_dir, env=cmd_env)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Training failed with exit code {e.returncode}", fg=typer.colors.RED, err=True)
        raise typer.Exit(e.returncode)
    

@app.command()
def test(
    project_dir: Path = typer.Argument(CURRENT_DIR, help="Path to the GeoSave project directory (containing geosave.toml)"),
    artifacts: str | None = typer.Option(None, "--artifacts", "-a", help="Path to a specific model artifacts directory to evaluate against"),
    env_vars: list[str] | None = typer.Option(None, "--env", "-e", help="Set environment variable(s) for the script (format: KEY=VALUE)")
):
    """
    Test models evaluated under a GeoSave pipeline.
    
    This executes the testing logic using your generated `train.py test` script. 
    It requires selecting an artifact directory inside `artifacts/` that contains 
    saved model checkpoints and configuration files. If `--artifacts` isn't found, 
    the CLI prompts you interactively.
    """
    workspace_config = validate_workspace(project_dir)
    test_script = project_dir / "test.py"

    if not test_script.exists():
        typer.secho(f"Error: test.py not found in {project_dir}", fg=typer.colors.RED, err=True)
        return
        
    run_args = _get_artifacts_args(project_dir, artifacts)
    cmd_env = _build_env(project_dir, env_vars)
    args_str = " ".join(run_args) if run_args else "No extra args"
    typer.secho(f"Starting test for '{workspace_config.get('project_name', 'Unknown')}'\nExecuting: `python test.py {args_str}`", fg=typer.colors.GREEN)
    
    try:
        subprocess.run(["python", str(test_script.resolve())] + run_args, check=True, cwd=project_dir, env=cmd_env)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Test failed with exit code {e.returncode}", fg=typer.colors.RED, err=True)
        raise typer.Exit(e.returncode)


@app.command()
def infer(
    project_dir: Path = typer.Argument(CURRENT_DIR, help="Path to the GeoSave project directory (containing geosave.toml)"),
    artifacts: str | None = typer.Option(None, "--artifacts", "-a", help="Path to the trained model artifacts directory used for predictions"),
    env_vars: list[str] | None = typer.Option(None, "--env", "-e", help="Set environment variable(s) for the script (format: KEY=VALUE)")
):
    """
    Run predictions inside a GeoSave workspace.
    
    This command invokes `inference.py` inside the project folder. It requires specifying 
    which artifact folder (containing your trained weights and config metadata) to use 
    to properly reconstruct the model before testing your data.
    """
    workspace_config = validate_workspace(project_dir)
    infer_script = project_dir / "inference.py"

    if not infer_script.exists():
        typer.secho(f"Error: inference.py not found in {project_dir}", fg=typer.colors.RED, err=True)
        return
        
    run_args = _get_artifacts_args(project_dir, artifacts)
    cmd_env = _build_env(project_dir, env_vars)
    args_str = " ".join(run_args) if run_args else "No extra args"
    typer.secho(f"Starting inference for '{workspace_config.get('project_name', 'Unknown')}'\nExecuting: `python inference.py {args_str}`", fg=typer.colors.GREEN)
    
    try:
        subprocess.run(["python", str(infer_script.resolve())] + run_args, check=True, cwd=project_dir, env=cmd_env)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Inference failed with exit code {e.returncode}", fg=typer.colors.RED, err=True)
        raise typer.Exit(e.returncode)


if __name__ == "__main__":
    app()