from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from geosave_engine.cli.errors import AbortedByUserError, BuildError
from geosave_engine.cli.plugin.plugin import add_plugin


class FakePrompter:
    def __init__(
        self,
        *,
        select_mapping_answers: list[str] | None = None,
        confirm_answers: list[bool] | None = None,
    ) -> None:
        self._select_mapping_answers = list(select_mapping_answers or [])
        self._confirm_answers = list(confirm_answers or [])
        self.confirm_calls = 0

    def select(
        self,
        message: str,
        choices: Sequence[str],
        *,
        default: str | None = None,
    ) -> str | None:
        raise AssertionError("select() should not be called in these tests")

    def select_mapping(
        self,
        message: str,
        choices: Sequence[tuple[str, str]],
        *,
        default: str | None = None,
    ) -> str | None:
        if not self._select_mapping_answers:
            return None
        return self._select_mapping_answers.pop(0)

    def text(self, message: str, *, default: str | None = None) -> str | None:
        raise AssertionError("text() should not be called in these tests")

    def confirm(self, message: str, *, default: bool = False) -> bool:
        self.confirm_calls += 1
        if not self._confirm_answers:
            return default
        return self._confirm_answers.pop(0)


class FakeConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warn(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.messages.append(message)

    def success(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    project_dir = tmp_path / "wawa"
    project_dir.mkdir(parents=True)
    (project_dir / "geosave.toml").write_text(
        "\n".join(
            [
                "project_name = 'wawa'",
                "task = 'semantic_segmentation'",
                "method = 'supervised'",
                "description = 'test'",
                "models = ['dummy']",
            ]
        ),
        encoding="utf-8",
    )
    return project_dir


@pytest.fixture
def plugin_templates(tmp_path: Path) -> Path:
    root = tmp_path / "plugins"

    scripts = root / "scripts" / "dynamic_world_ingest"
    scripts.mkdir(parents=True)
    (scripts / "dw_download.py").write_text("print('download')\n", encoding="utf-8")

    notebook_dir = root / "notebook"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "exploratory_data_analysis.ipynb").write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
        encoding="utf-8",
    )

    return root


def test_add_scripts_plugin_with_explicit_args(
    workspace: Path,
    plugin_templates: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geosave_engine.cli.plugin.plugin.plugins_root", lambda: plugin_templates)
    prompter = FakePrompter()
    console = FakeConsole()

    add_plugin(
        workspace,
        plugin_type="script",
        plugin_name="dynamic_world_ingest",
        prompter=prompter,
        console=console,
    )

    assert (workspace / "scripts" / "dynamic_world_ingest" / "dw_download.py").is_file()


def test_add_notebook_plugin_copies_to_notebooks_dir(
    workspace: Path,
    plugin_templates: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geosave_engine.cli.plugin.plugin.plugins_root", lambda: plugin_templates)
    prompter = FakePrompter()
    console = FakeConsole()

    add_plugin(
        workspace,
        plugin_type="notebook",
        plugin_name="exploratory_data_analysis",
        prompter=prompter,
        console=console,
    )

    assert (workspace / "notebooks" / "exploratory_data_analysis.ipynb").is_file()


def test_add_plugin_with_interactive_prompts(
    workspace: Path,
    plugin_templates: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geosave_engine.cli.plugin.plugin.plugins_root", lambda: plugin_templates)
    prompter = FakePrompter(select_mapping_answers=["scripts", "dynamic_world_ingest"])
    console = FakeConsole()

    add_plugin(
        workspace,
        plugin_type=None,
        plugin_name=None,
        prompter=prompter,
        console=console,
    )

    assert (workspace / "scripts" / "dynamic_world_ingest" / "dw_download.py").is_file()


def test_add_plugin_rejects_unknown_plugin_type(
    workspace: Path,
    plugin_templates: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geosave_engine.cli.plugin.plugin.plugins_root", lambda: plugin_templates)

    with pytest.raises(BuildError):
        add_plugin(
            workspace,
            plugin_type="unknown",
            plugin_name="dynamic_world_ingest",
            prompter=FakePrompter(),
            console=FakeConsole(),
        )


def test_add_scripts_plugin_prompts_on_file_collision_and_can_cancel(
    workspace: Path,
    plugin_templates: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geosave_engine.cli.plugin.plugin.plugins_root", lambda: plugin_templates)
    existing = workspace / "scripts" / "dynamic_world_ingest"
    existing.mkdir(parents=True)
    (existing / "dw_download.py").write_text("print('old')\n", encoding="utf-8")

    prompter = FakePrompter(confirm_answers=[False])
    with pytest.raises(AbortedByUserError):
        add_plugin(
            workspace,
            plugin_type="scripts",
            plugin_name="dynamic_world_ingest",
            prompter=prompter,
            console=FakeConsole(),
        )

    assert prompter.confirm_calls == 1
    assert (existing / "dw_download.py").read_text(encoding="utf-8") == "print('old')\n"


def test_add_scripts_plugin_does_not_prompt_when_no_file_collision(
    workspace: Path,
    plugin_templates: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geosave_engine.cli.plugin.plugin.plugins_root", lambda: plugin_templates)
    # Existing directory should not trigger prompt by itself when no files will be overwritten.
    (workspace / "scripts" / "dynamic_world_ingest").mkdir(parents=True)

    # Replace template with a non-colliding file for this test case.
    template_dir = plugin_templates / "scripts" / "dynamic_world_ingest"
    (template_dir / "dw_download.py").unlink()
    (template_dir / "dw_extract.py").write_text("print('extract')\n", encoding="utf-8")

    prompter = FakePrompter()
    add_plugin(
        workspace,
        plugin_type="scripts",
        plugin_name="dynamic_world_ingest",
        prompter=prompter,
        console=FakeConsole(),
    )

    assert prompter.confirm_calls == 0
    assert (workspace / "scripts" / "dynamic_world_ingest" / "dw_extract.py").is_file()
