from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import questionary

from geosave_engine.cli.errors import AbortedByUserError, WorkspaceError

if TYPE_CHECKING:
    from .model import Workspace

_CONFIG_FILE = "config.yaml"
_CHECKPOINTS_DIR = "checkpoints"
_CHECKPOINT_GLOB = "*.ckpt"


@dataclass(frozen=True)
class RunArtifact:
    """Store paths and identity for one model run.

    Args:
        run_dir: Version directory containing artifacts for one run
            (for example, artifacts/DynamicWorld/version_9).
        config_path: Lightning config saved for the run.
        checkpoint_paths: All discovered checkpoints for the run, sorted.
    """

    run_dir: Path
    config_path: Path
    checkpoint_paths: list[Path]

    @property
    def run_name(self) -> str:
        """Return the run's parent directory name (for example, "DynamicWorld")."""
        return self.run_dir.parent.name


def discover_artifacts(artifacts_dir: Path) -> list[Path]:
    """Find run version directories without reading their contents.

    Args:
        artifacts_dir: Workspace artifacts directory.

    Returns:
        Sorted version directories (artifacts/<run_name>/version_N) that
        hold a config.yaml.
    """
    if not artifacts_dir.is_dir():
        return []

    return sorted(
        path.resolve()
        for path in artifacts_dir.glob("*/version_*")
        if path.is_dir() and (path / _CONFIG_FILE).is_file()
    )


def load_run_artifact(run_dir: Path) -> RunArtifact:
    """Load one run directory using the canonical artifact layout.

    Args:
        run_dir: Version directory containing one model run.

    Returns:
        Validated paths for the run.

    Raises:
        WorkspaceError: If required files are missing.
    """
    resolved_run_dir = run_dir.expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise WorkspaceError(f"Artifact run directory not found: {resolved_run_dir}")

    config_path = _require_file(resolved_run_dir / _CONFIG_FILE)
    checkpoint_paths = _discover_checkpoints(resolved_run_dir / _CHECKPOINTS_DIR)

    return RunArtifact(
        run_dir=resolved_run_dir,
        config_path=config_path,
        checkpoint_paths=checkpoint_paths,
    )


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise WorkspaceError(f"Required artifact file not found: {path}")
    return path.resolve()


def _discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.is_dir():
        raise WorkspaceError(f"Checkpoints directory not found: {checkpoints_dir}")

    checkpoint_paths = sorted(path.resolve() for path in checkpoints_dir.glob(_CHECKPOINT_GLOB))
    if not checkpoint_paths:
        raise WorkspaceError(f"No checkpoint files found in: {checkpoints_dir}")
    return checkpoint_paths


def artifact_paths(workspace: Workspace) -> dict[str, Path]:
    """Map artifact keys to their run directories.

    Args:
        workspace: Loaded workspace to scan.

    Returns:
        Keys like "model_name/version_0" mapped to their run directory.
    """
    return {
        str(path.relative_to(workspace.artifacts_dir)): path
        for path in workspace.artifacts
    }


def resolve_artifact_name(workspace: Workspace, artifact_name: str | None) -> str:
    """Resolve one artifact key, prompting when omitted.

    Args:
        workspace: Loaded workspace to scan.
        artifact_name: Explicit artifact key, skipping the prompt.

    Returns:
        Validated artifact key, usable with artifact_paths(workspace).

    Raises:
        WorkspaceError: If no artifacts exist, or artifact_name doesn't
            match any.
        AbortedByUserError: If the prompt is cancelled.
    """
    paths = artifact_paths(workspace)
    if artifact_name is None:
        if not paths:
            raise WorkspaceError(f"No artifacts found in: {workspace.artifacts_dir}")
        artifact_name = _prompt_for_artifact(list(paths))

    if artifact_name not in paths:
        available = ", ".join(sorted(paths))
        raise WorkspaceError(f"Artifact not found: {artifact_name}. Available: {available}")

    return artifact_name


def select_checkpoint(artifact: RunArtifact, checkpoint: str | None) -> Path:
    """Resolve one checkpoint from a loaded RunArtifact, prompting if ambiguous.

    Args:
        artifact: Loaded run artifact with all discovered checkpoint_paths.
        checkpoint: Checkpoint filename to use directly, skipping the
            prompt. Prompt only when multiple checkpoints exist.

    Returns:
        Resolved checkpoint path.

    Raises:
        WorkspaceError: If checkpoint doesn't match any discovered file.
        AbortedByUserError: If the prompt is cancelled.
    """
    checkpoint_paths = artifact.checkpoint_paths

    if checkpoint is not None:
        matches = [path for path in checkpoint_paths if path.name == checkpoint]
        if not matches:
            available = ", ".join(path.name for path in checkpoint_paths)
            raise WorkspaceError(f"Checkpoint not found: {checkpoint}. Available: {available}")
        return matches[0]

    if len(checkpoint_paths) == 1:
        return checkpoint_paths[0]

    answer = questionary.select(
        "Select a checkpoint:",
        choices=[path.name for path in checkpoint_paths],
    ).ask()
    if answer is None:
        raise AbortedByUserError("Checkpoint selection was aborted by the user.")

    return next(path for path in checkpoint_paths if path.name == answer.strip())


def _prompt_for_artifact(artifact_keys: list[str]) -> str:
    answer = questionary.select("Select an artifact:", choices=artifact_keys).ask()
    if answer is None:
        raise AbortedByUserError("Artifact selection was aborted by the user.")
    return answer.strip()
