from __future__ import annotations

from pathlib import Path

import pytest
import toml

from geosave_engine.cli.errors import WorkspaceError
from geosave_engine.cli.runtime.workspace import Workspace, announce_workspace, load_workspace

from tests.fakes import StubConsole


def _write_workspace(root: Path, payload: dict) -> None:
    with open(root / "geosave.toml", "w", encoding="utf-8") as handle:
        toml.dump(payload, handle)


def test_load_workspace_parses_expected_fields(tmp_path: Path) -> None:
    _write_workspace(
        tmp_path,
        {
            "project_name": "demo",
            "task": "semantic segmentation",
            "method": "supervised",
            "description": "a sample",
            "models": ["ModelA", "ModelB"],
        },
    )

    workspace = load_workspace(tmp_path)

    assert isinstance(workspace, Workspace)
    assert workspace.project_name == "demo"
    assert workspace.task == "semantic segmentation"
    assert workspace.method == "supervised"
    assert workspace.models == ["ModelA", "ModelB"]
    assert workspace.root == tmp_path


def test_load_workspace_fills_defaults_for_missing_fields(tmp_path: Path) -> None:
    _write_workspace(tmp_path, {"project_name": "partial"})

    workspace = load_workspace(tmp_path)

    assert workspace.project_name == "partial"
    assert workspace.task == ""
    assert workspace.method == ""
    assert workspace.description == ""
    assert workspace.models == []


def test_load_workspace_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        load_workspace(tmp_path)


def test_announce_workspace_routes_through_console(tmp_path: Path) -> None:
    _write_workspace(tmp_path, {"project_name": "demo"})
    workspace = load_workspace(tmp_path)
    console = StubConsole()

    announce_workspace(workspace, console)

    assert console.messages == [("info", "Found GeoSave project workspace: 'demo'")]
