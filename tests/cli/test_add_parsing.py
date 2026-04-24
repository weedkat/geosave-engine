from __future__ import annotations

from pathlib import Path

from geosave_engine.cli.main import _normalize_add_positionals


def test_normalize_add_positionals_uses_current_workspace_for_short_form(
    monkeypatch,
    tmp_path: Path,
) -> None:
    current_workspace = tmp_path / "wawa"
    current_workspace.mkdir()
    monkeypatch.setattr("geosave_engine.cli.main.CURRENT_DIR", current_workspace)

    resolved_project, resolved_type, resolved_name = _normalize_add_positionals(
        Path("script"),
        "dynamic_world_ingest",
        None,
    )

    assert resolved_project == current_workspace
    assert resolved_type == "script"
    assert resolved_name == "dynamic_world_ingest"


def test_normalize_add_positionals_keeps_explicit_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "my_project"
    workspace.mkdir()
    (workspace / "geosave.toml").write_text("project_name = 'x'\ntask = 'y'\nmethod = 'z'\n", encoding="utf-8")

    resolved_project, resolved_type, resolved_name = _normalize_add_positionals(
        workspace,
        "scripts",
        "dynamic_world_ingest",
    )

    assert resolved_project == workspace
    assert resolved_type == "scripts"
    assert resolved_name == "dynamic_world_ingest"
