from __future__ import annotations

from pathlib import Path

import toml
from typer.testing import CliRunner

from geosave_engine.cli.main import app
from geosave_engine.cli.workspace import RunArtifact, Workspace, WorkspaceSpec
from geosave_engine.cli.workspace import scaffold as scaffold_module


runner = CliRunner()


def test_main_lists_command_modules() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "create" in result.stdout
    assert "upload" in result.stdout
    assert "infra" in result.stdout


def test_workspace_scaffold_uses_bundled_templates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    templates_dir = tmp_path / "bundled-templates"
    common_dir = templates_dir / "common"
    method_dir = templates_dir / "semantic_segmentation" / "supervised"
    common_dir.mkdir(parents=True)
    method_dir.mkdir(parents=True)
    (common_dir / "main.py").write_text("# workspace entry point\n")
    (method_dir / "configs").mkdir()
    (method_dir / "configs" / "model.yaml").write_text("model: {}\n")

    monkeypatch.setattr(scaffold_module, "templates_dir", lambda: templates_dir)
    monkeypatch.setattr(scaffold_module, "common_template_dir", lambda: common_dir)

    spec = WorkspaceSpec(
        project_name="demo",
        project_task="semantic_segmentation",
        project_method="supervised",
    )
    workspace = Workspace(tmp_path, spec)

    workspace.setup_workspace()

    assert (workspace.root / "main.py").exists()
    assert (workspace.root / "configs" / "model.yaml").exists()
    assert toml.load(workspace.root / "geosave.toml")["project_name"] == "demo"


def test_run_artifact_uses_parent_directory_name(tmp_path: Path) -> None:
    artifact = RunArtifact(
        run_dir=tmp_path / "artifacts" / "forest-run" / "version_0",
        config_path=tmp_path / "config.yaml",
        checkpoint_paths=[tmp_path / "checkpoints" / "last.ckpt"],
    )

    assert artifact.model_name == "forest-run"
