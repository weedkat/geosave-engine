from __future__ import annotations

from pathlib import Path

import pytest

from geosave_engine.cli.build import BuildRequest, collect_build_request, generate_project
from geosave_engine.cli.errors import AbortedByUser, BuildError
from geosave_engine.cli.paths import models_root, templates_root

from tests.fakes import FakePrompter, StubConsole


def _real_template_dir() -> Path:
    return templates_root()


def _real_models_dir() -> Path:
    return models_root()


def test_collect_build_request_composes_from_prompter_answers() -> None:
    prompter = FakePrompter(
        selects=["semantic segmentation", "supervised", "DensePredictionTransformer"],
        texts=["myproject", "hello"],
    )
    console = StubConsole()

    request = collect_build_request(
        None,
        template_dir=_real_template_dir(),
        models_package_path=_real_models_dir(),
        prompter=prompter,
        console=console,
    )

    assert isinstance(request, BuildRequest)
    assert request.name == "myproject"
    assert request.task == "semantic segmentation"
    assert request.method == "supervised"
    assert request.selected_model == "DensePredictionTransformer"
    assert request.description == "hello"
    # Project name is answered via `text`, not `select`, so no name prompt when pre-seeded:
    assert [call[0] for call in prompter.calls] == ["text", "select", "select", "select", "text"]


def test_collect_build_request_uses_initial_name_and_skips_name_prompt() -> None:
    prompter = FakePrompter(
        selects=["semantic segmentation", "supervised", "DensePredictionTransformer"],
        texts=["desc"],
    )
    console = StubConsole()

    request = collect_build_request(
        "preseeded",
        template_dir=_real_template_dir(),
        models_package_path=_real_models_dir(),
        prompter=prompter,
        console=console,
    )

    assert request.name == "preseeded"
    # No name prompt was issued since we supplied an initial value.
    assert [call[0] for call in prompter.calls] == ["select", "select", "select", "text"]


def test_collect_build_request_raises_when_template_dir_empty(tmp_path: Path) -> None:
    prompter = FakePrompter()
    console = StubConsole()

    with pytest.raises(BuildError):
        collect_build_request(
            "x",
            template_dir=tmp_path,
            models_package_path=_real_models_dir(),
            prompter=prompter,
            console=console,
        )


def test_generate_project_writes_workspace(tmp_path: Path) -> None:
    prompter = FakePrompter()
    console = StubConsole()
    request = BuildRequest(
        name="demo",
        task="semantic segmentation",
        method="supervised",
        available_models=["DensePredictionTransformer"],
        selected_model="DensePredictionTransformer",
        description="smoke",
    )

    project = generate_project(
        request,
        output_dir=tmp_path,
        template_dir=_real_template_dir(),
        prompter=prompter,
        console=console,
    )

    assert project == tmp_path / "demo"
    assert (project / "geosave.toml").is_file()
    assert (project / "configs" / "default.yaml").is_file()
    assert (project / "src" / "lightning_module.py").is_file()
    # Success should have been announced on the console.
    assert any(level == "success" for level, _ in console.messages)


def test_generate_project_aborts_when_user_declines_overwrite(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    prompter = FakePrompter(confirms=[False])
    console = StubConsole()
    request = BuildRequest(
        name="demo",
        task="semantic segmentation",
        method="supervised",
        available_models=["DensePredictionTransformer"],
        selected_model="DensePredictionTransformer",
        description="smoke",
    )

    with pytest.raises(AbortedByUser):
        generate_project(
            request,
            output_dir=tmp_path,
            template_dir=_real_template_dir(),
            prompter=prompter,
            console=console,
        )
