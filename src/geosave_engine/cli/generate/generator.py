from __future__ import annotations

from pathlib import Path

import toml

from geosave_engine.cli.generate.request import BuildRequest
from geosave_engine.cli.errors import AbortedByUser, BuildError
from geosave_engine.cli.io import Console, Prompter
from geosave_engine.utils.cli.fs_ops import copy_tree
from geosave_engine.utils.ml.yaml_config import inject_into_file


def generate_project(
    request: BuildRequest,
    output_dir: Path,
    *,
    template_dir: Path,
    prompter: Prompter,
    console: Console,
) -> Path:
    """Materialize `request` into a workspace under `output_dir / request.name`.

    Returns the absolute path to the generated project. Raises `BuildError`
    on copy/template errors and `AbortedByUser` if the user declines an
    overwrite prompt.
    """
    project_dir = output_dir / request.name

    _prompt_overwrite_if_exists(project_dir, prompter)

    _copy_common_template(template_dir, project_dir)
    # _copy_task_common(template_dir, request.task, project_dir)
    _overlay_task_template(template_dir, request.task, request.method, project_dir)
    _write_project_metadata(project_dir, request)
    _inject_selected_model(project_dir, request.selected_model)

    console.success(f"Project '{request.name}' created successfully at {project_dir}")
    return project_dir


def _prompt_overwrite_if_exists(project_dir: Path, prompter: Prompter) -> None:
    if not project_dir.exists():
        return
    proceed = prompter.confirm(
        f"Destination '{project_dir}' already exists. Overwrite?",
        default=False,
    )
    if not proceed:
        raise AbortedByUser("Build cancelled — destination exists.")


def _copy_common_template(template_dir: Path, project_dir: Path) -> None:
    common = template_dir / "common"
    try:
        copy_tree(common, project_dir, file_exceptions=["*/.gitkeep", "*/.ruff_cache"])
    except Exception as error:
        raise BuildError(f"Failed to copy common template: {error}") from error


def _copy_task_common(template_dir: Path, task: str, project_dir: Path) -> None:
    task_slug = task.replace(" ", "_")
    task_common = template_dir / task_slug / "common"
    if not task_common.exists():
        return
    try:
        copy_tree(task_common, project_dir, file_exceptions=["*/.gitkeep", "*/.ruff_cache"])
    except Exception as error:
        raise BuildError(f"Failed to copy task common template: {error}") from error


def _overlay_task_template(
    template_dir: Path, task: str, method: str, project_dir: Path
) -> None:
    task_slug = task.replace(" ", "_")
    method_slug = method.replace(" ", "_")
    source = template_dir / task_slug / method_slug
    try:
        copy_tree(source, project_dir, file_exceptions=["*/.gitkeep", "*/.ruff_cache"])
    except Exception as error:
        raise BuildError(f"Failed to overlay task template: {error}") from error


def _write_project_metadata(project_dir: Path, request: BuildRequest) -> None:
    metadata = {
        "project_name": request.name,
        "task": request.task,
        "method": request.method,
        "models": request.available_models,
        "description": request.description,
    }
    try:
        with open(project_dir / "geosave.toml", "w", encoding="utf-8") as handle:
            toml.dump(metadata, handle)
    except Exception as error:
        raise BuildError(f"Failed to write geosave.toml: {error}") from error


def _inject_selected_model(project_dir: Path, selected_model: str) -> None:
    config_path = project_dir / "configs" / "default.yaml"
    try:
        inject_into_file(
            config_path,
            "model.name",
            selected_model,
            required_mapping_keys=("model",),
        )
    except Exception as error:
        raise BuildError(f"Failed to inject selected model into config: {error}") from error
