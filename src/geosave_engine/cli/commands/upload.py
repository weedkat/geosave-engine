from __future__ import annotations

import importlib
import os
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
import typer
import yaml

from geosave_engine.cli.errors import AbortedByUserError, WorkspaceError

if TYPE_CHECKING:
    # Only ever used as type annotations (lazy strings, thanks to
    # `from __future__ import annotations`) — never called by name at
    # runtime, so keep them out of the module's real import graph. Both
    # drag in ~10s of lightning.pytorch/mlflow import time, which main.py
    # would otherwise pay on every `geosave` invocation (create, artifact,
    # --help, ...), not just `upload`.
    from lightning.pytorch import LightningModule
    from mlflow.models.model import ModelInfo
from geosave_engine.cli.workspace import Workspace, load_run_artifact
from geosave_engine.cli.workspace.artifact import (
    artifact_paths,
    resolve_artifact_name,
    select_checkpoint,
)


def upload(
    project_dir: Path | None = typer.Argument(None, help="Workspace directory."),
    artifact_name: str | None = typer.Option(
        None,
        "--artifact",
        "-a",
        help="Artifact directory (for example, model_name/version_0).",
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Checkpoint filename inside the run's checkpoints/ dir. Prompt if omitted and multiple exist.",
    ),
    registered_model_name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="MLflow registered model name. Defaults to the resolved run name.",
    ),
) -> None:
    """Upload one trained checkpoint to the MLflow model registry.

    Resolve one run artifact (discover it under artifacts/, prompt if
    ambiguous), rebuild its LightningModule from saved config + checkpoint
    weights, then
    log + register it to MLflow inside a fresh mlflow.start_run(), with
    workspace modules/ bundled as code.

    Args:
        project_dir: Workspace directory. Defaults to cwd.
        artifact_name: Artifact directory relative to artifacts/, for
            example "DynamicWorld/version_9". Prompt if omitted.
        checkpoint: Checkpoint filename. Prompt if omitted and multiple
            checkpoints exist.
        registered_model_name: MLflow registered model name. Defaults to
            the resolved RunArtifact.model_name.

    Raises:
        WorkspaceError: If artifact/config is missing or invalid.
        AbortedByUserError: If prompted for a missing MLFLOW_TRACKING_URI or
            MLFLOW_EXPERIMENT_NAME and no answer is given.
    """
    workspace = Workspace.load_workspace(project_dir or Path.cwd())

    # modules/ has no __init__.py (namespace package) — only resolves once
    # workspace.root is searchable. python main.py gets this for free from
    # cwd; this installed console script doesn't, so make it explicit.
    # No-op for premade (Path A) classes — those come from the installed
    # geosave_engine package and resolve regardless.
    sys.path.insert(0, str(workspace.root))

    resolved_name = resolve_artifact_name(workspace, artifact_name)
    run_dir = artifact_paths(workspace)[resolved_name]
    artifact = load_run_artifact(run_dir)

    checkpoint_path = select_checkpoint(artifact, checkpoint)
    model = _load_model_from_checkpoint(artifact.config_path, checkpoint_path)

    resolved_registered_name = registered_model_name or artifact.model_name
    tracking_uri = _require_tracking_uri()
    experiment_name = _require_experiment_name(resolved_registered_name)

    model_info = log_model(
        model=model,
        name=resolved_registered_name,
        checkpoint_path=checkpoint_path,
        modules_dir=workspace.modules_dir,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    # model_info.model_uri is mlflow 3.x's models:/m-<hash> model-id form —
    # print the conventional name/version form instead, since that's what
    # litserve configs and humans actually reference.
    typer.echo(f"models:/{resolved_registered_name}/{model_info.registered_model_version}")


def _import_model_class(class_path: str) -> type[LightningModule]:
    """Dynamically import a LightningModule class from its dotted path.

    Args:
        class_path: Fully qualified class path (for example,
            "geosave_engine.ml.tasks.SemanticSegmentationTask"), read from
            the run's config.yaml "model.class_path".

    Returns:
        Imported class, ready for load_from_checkpoint.

    Raises:
        WorkspaceError: If the module or class can't be imported.
    """
    module_name, _, class_name = class_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise WorkspaceError(f"Could not import model class '{class_path}': {error}") from error


def _load_model_from_checkpoint(config_path: Path, checkpoint_path: Path) -> LightningModule:
    """Rebuild one trained LightningModule from its run artifacts.

    Uses the checkpoint's saved hyperparameters (save_hyperparameters() at
    training time) — no need to re-pass init_args from config.yaml.

    Args:
        config_path: Run's config.yaml, used only to resolve the model
            class_path.
        checkpoint_path: Checkpoint file with saved weights + hparams.

    Returns:
        Instantiated model with weights loaded, ready to log.

    Raises:
        WorkspaceError: If config.yaml lacks a "model.class_path" entry, or
            the checkpoint file is unreadable or corrupted.
    """
    from geosave_engine.ml.inference.protocol import Predictable

    config = yaml.safe_load(config_path.read_text())
    class_path = config.get("model", {}).get("class_path")
    if not class_path:
        raise WorkspaceError(f"config.yaml missing model.class_path: {config_path}")

    model_cls = _import_model_class(class_path)
    try:
        model = model_cls.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    except (RuntimeError, OSError, EOFError, pickle.UnpicklingError) as error:
        raise WorkspaceError(f"Could not load checkpoint '{checkpoint_path.name}': {error}") from error

    if not isinstance(model, Predictable):
        raise WorkspaceError(
            f"{class_path} doesn't implement Predictable (no usable predict() method) — "
            "can't register a model that can't be served."
        )
    return model


def _require_tracking_uri() -> str:
    """Read MLFLOW_TRACKING_URI from the environment, prompting when unset.

    No local-mlruns fallback — upload always targets a real registry, so an
    empty answer aborts instead of silently defaulting.

    Returns:
        Tracking URI value.

    Raises:
        AbortedByUserError: If prompted and the user gives no answer.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri

    answer = questionary.text(
        "MLFLOW_TRACKING_URI is not set. Enter your MLflow tracking URI:"
    ).ask()
    if not answer or not answer.strip():
        raise AbortedByUserError("MLflow tracking URI is required.")
    return answer.strip()


def _require_experiment_name(default: str) -> str:
    """Read MLFLOW_EXPERIMENT_NAME from the environment, prompting when unset.

    Mirrors GeosaveCLI's own training-time default (falls back to
    model_name when MLFLOW_EXPERIMENT_NAME isn't set — see ml/cli/cli.py).

    Args:
        default: Prefilled answer, typically the resolved model name.

    Returns:
        Experiment name value.

    Raises:
        AbortedByUserError: If prompted and the user gives no answer.
    """
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if experiment_name:
        return experiment_name

    answer = questionary.text(
        "MLFLOW_EXPERIMENT_NAME is not set. Enter an experiment name:",
        default=default,
    ).ask()
    if not answer or not answer.strip():
        raise AbortedByUserError("MLflow experiment name is required.")
    return answer.strip()


def log_model(
    model: LightningModule,
    name: str,
    checkpoint_path: Path,
    modules_dir: Path,
    tracking_uri: str,
    experiment_name: str,
) -> ModelInfo:
    """Log one model run to MLflow, resolving tracking uri/experiment/run itself.

    MLflow acts as registry using pytorch log_model. modules_dir is
    bundled as code_paths so registry-side consumers (litserve) can import
    workspace-local preprocessing (for example, modules.data_pipeline) after
    pulling the model; litserve owns preprocessing + inference wiring, not
    the logged artifact.

    Args:
        model: Instantiated model to upload.
        name: Resolved name (for example, "DynamicWorld") — used as the
            MLflow run name, the artifact's name inside it, and the
            registered model name. upload only exposes one --name flag, so
            these three never differ in practice.
        checkpoint_path: Local checkpoint this model was rebuilt from,
            stored as metadata for traceability.
        modules_dir: Workspace modules/ directory bundled as code_paths —
            code MLflow copies into the model's code/ dir and prepends to
            sys.path on load.
        tracking_uri: MLflow tracking URI to log against.
        experiment_name: MLflow experiment to log under.

    Returns:
        Logged MLflow model information.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    typer.echo(f"Uploading '{name}' ({checkpoint_path.name}) to {tracking_uri} ...", err=True)
    with mlflow.start_run(run_name=name):
        # MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR defaults on — mlflow shows
        # its own tqdm byte-level progress for the artifact upload itself.
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name=name,
            code_paths=[str(modules_dir)],
            registered_model_name=name,
            metadata={"checkpoint": checkpoint_path.name},
        )
    typer.echo(f"Registered '{name}' version {model_info.registered_model_version}", err=True)

    return model_info
