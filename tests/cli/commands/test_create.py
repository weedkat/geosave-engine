"""Workspace selection and metadata through the public CLI."""

import tomllib

import pytest
from typer.testing import CliRunner

from geosave_engine.cli.main import app


@pytest.mark.parametrize("workspace", ["segmentation", "custom_lightning", "blank"])
def test_create_selects_workspace(tmp_path, monkeypatch, workspace):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        ["create", "demo", "--description", "Test project", "--workspace", workspace],
    )
    assert result.exit_code == 0, result.output
    root = tmp_path / "demo"
    assert (root / "main.py").is_file()
    assert (root / ".env").is_file()
    assert not (root / "description.txt").exists()
    metadata = tomllib.loads((root / "geosave.toml").read_text())
    assert "task" not in metadata["workspace"]
    assert "method" not in metadata["workspace"]
    if workspace == "blank":
        assert metadata["workspace"] == {}
        assert not (root / "configs/train.yaml").exists()
    else:
        assert metadata["workspace"]["template"] == workspace
        assert (root / "configs/train.yaml").is_file()


def test_create_rejects_unknown_workspace_before_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["create", "demo", "-d", "Test", "--workspace", "missing"]
    )
    assert result.exit_code == 2
    assert "not a valid workspace" in result.output
    assert not (tmp_path / "demo").exists()


def test_make_installs_scaffold_in_existing_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "geosave.toml").write_text("[workspace]\n")
    result = CliRunner().invoke(app, ["make", "scripts", "prepare_dense_data.py"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "scripts/prepare_dense_data.py").is_file()
