from __future__ import annotations

from pathlib import Path

import pytest

from geosave_engine.cli.errors import AbortedByUserError, WorkspaceError
from geosave_engine.cli.workspace import Workspace, WorkspaceSpec, discover_artifacts, load_run_artifact
from geosave_engine.cli.workspace import artifact as artifact_module
from geosave_engine.cli.workspace.artifact import artifact_paths, resolve_artifact_name, select_checkpoint


def _write_run(run_dir: Path, checkpoint_names: list[str]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model: {}\n")
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir()
    for name in checkpoint_names:
        (checkpoints_dir / name).write_bytes(b"checkpoint")


def test_discover_artifacts_returns_version_directories(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_9"
    _write_run(run_dir, ["last.ckpt"])
    # A version dir with no config.yaml isn't a discoverable run.
    (tmp_path / "artifacts" / "DynamicWorld" / "version_incomplete").mkdir(parents=True)

    artifacts = discover_artifacts(tmp_path / "artifacts")

    assert artifacts == [run_dir.resolve()]


def test_load_run_artifact_reads_canonical_run_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_9"
    _write_run(run_dir, ["last.ckpt", "epoch=00-val_loss=1.0000.ckpt"])

    artifact = load_run_artifact(run_dir)

    assert artifact.run_dir == run_dir.resolve()
    assert artifact.config_path == (run_dir / "config.yaml").resolve()
    assert {path.name for path in artifact.checkpoint_paths} == {
        "last.ckpt",
        "epoch=00-val_loss=1.0000.ckpt",
    }
    assert artifact.model_name == "DynamicWorld"


def test_load_run_artifact_requires_checkpoints_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_9"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model: {}\n")

    with pytest.raises(WorkspaceError, match="Checkpoints directory not found"):
        load_run_artifact(run_dir)


def _make_workspace(root: Path, run_keys: list[str]) -> Workspace:
    """Build a real Workspace over a root with the given artifact run dirs pre-created."""
    for key in run_keys:
        _write_run(root / "artifacts" / key, ["last.ckpt"])
    spec = WorkspaceSpec(
        project_name="demo",
        project_task="semantic_segmentation",
        project_method="supervised",
    )
    (root / "geosave.toml").touch()  # load_workspace() only needs it to exist here
    return Workspace(root, spec)


# ── artifact_paths / resolve_artifact_name ────────────────────────────────


def test_artifact_paths_keys_are_relative_to_artifacts_dir(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path, ["DynamicWorld/version_0", "DynamicWorld/version_1"])

    paths = artifact_paths(workspace)

    assert set(paths) == {"DynamicWorld/version_0", "DynamicWorld/version_1"}


def test_resolve_artifact_name_returns_explicit_name(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path, ["DynamicWorld/version_0"])

    assert resolve_artifact_name(workspace, "DynamicWorld/version_0") == "DynamicWorld/version_0"


def test_resolve_artifact_name_raises_when_no_artifacts_exist(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path, [])

    with pytest.raises(WorkspaceError, match="No artifacts found"):
        resolve_artifact_name(workspace, None)


def test_resolve_artifact_name_raises_on_unknown_name(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path, ["DynamicWorld/version_0"])

    with pytest.raises(WorkspaceError, match="Artifact not found"):
        resolve_artifact_name(workspace, "NotARun/version_0")


def test_resolve_artifact_name_prompts_when_omitted_and_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_workspace(tmp_path, ["DynamicWorld/version_0", "DynamicWorld/version_1"])
    seen_choices: list[list[str]] = []

    monkeypatch.setattr(
        artifact_module.questionary,
        "select",
        lambda _message, choices: _record_and_answer(seen_choices, choices, "DynamicWorld/version_1"),
    )

    assert resolve_artifact_name(workspace, None) == "DynamicWorld/version_1"
    assert set(seen_choices[0]) == {"DynamicWorld/version_0", "DynamicWorld/version_1"}


def test_resolve_artifact_name_aborts_when_prompt_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_workspace(tmp_path, ["DynamicWorld/version_0", "DynamicWorld/version_1"])

    monkeypatch.setattr(
        artifact_module.questionary,
        "select",
        lambda _message, choices: _record_and_answer([], choices, None),
    )

    with pytest.raises(AbortedByUserError):
        resolve_artifact_name(workspace, None)


# ── select_checkpoint ──────────────────────────────────────────────────────


def test_select_checkpoint_returns_explicit_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_0"
    _write_run(run_dir, ["last.ckpt", "epoch=00-val_loss=1.0000.ckpt"])
    artifact = load_run_artifact(run_dir)

    result = select_checkpoint(artifact, "epoch=00-val_loss=1.0000.ckpt")

    assert result.name == "epoch=00-val_loss=1.0000.ckpt"


def test_select_checkpoint_raises_on_unknown_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_0"
    _write_run(run_dir, ["last.ckpt"])
    artifact = load_run_artifact(run_dir)

    with pytest.raises(WorkspaceError, match="Checkpoint not found"):
        select_checkpoint(artifact, "does-not-exist.ckpt")


def test_select_checkpoint_returns_single_checkpoint_without_prompt(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_0"
    _write_run(run_dir, ["last.ckpt"])
    artifact = load_run_artifact(run_dir)

    assert select_checkpoint(artifact, None).name == "last.ckpt"


def test_select_checkpoint_prompts_when_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_0"
    _write_run(run_dir, ["last.ckpt", "epoch=00-val_loss=1.0000.ckpt"])
    artifact = load_run_artifact(run_dir)
    seen_choices: list[list[str]] = []

    monkeypatch.setattr(
        artifact_module.questionary,
        "select",
        lambda _message, choices: _record_and_answer(
            seen_choices, choices, "epoch=00-val_loss=1.0000.ckpt"
        ),
    )

    result = select_checkpoint(artifact, None)

    assert result.name == "epoch=00-val_loss=1.0000.ckpt"
    assert set(seen_choices[0]) == {"last.ckpt", "epoch=00-val_loss=1.0000.ckpt"}


def test_select_checkpoint_aborts_when_prompt_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "artifacts" / "DynamicWorld" / "version_0"
    _write_run(run_dir, ["last.ckpt", "epoch=00-val_loss=1.0000.ckpt"])
    artifact = load_run_artifact(run_dir)

    monkeypatch.setattr(
        artifact_module.questionary,
        "select",
        lambda _message, choices: _record_and_answer([], choices, None),
    )

    with pytest.raises(AbortedByUserError):
        select_checkpoint(artifact, None)


class _FakeSelect:
    def __init__(self, answer: str | None) -> None:
        self._answer = answer

    def ask(self) -> str | None:
        return self._answer


def _record_and_answer(seen_choices: list[list[str]], choices: list[str], answer: str | None) -> _FakeSelect:
    seen_choices.append(choices)
    return _FakeSelect(answer)
