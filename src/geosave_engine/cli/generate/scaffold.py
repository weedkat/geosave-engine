from __future__ import annotations

from pathlib import Path

from geosave_engine.cli.generate.request import BuildRequest
from geosave_engine.cli.errors import AbortedByUser, BuildError
from geosave_engine.cli.io import Console, Prompter
from geosave_engine.cli.search import discover_model_names, discover_tasks


def collect_build_request(
    initial_name: str | None,
    *,
    template_dir: Path,
    models_package_path: Path,
    prompter: Prompter,
    console: Console,
) -> BuildRequest:
    """Interactively build a `BuildRequest` by prompting the user.

    Pure composition over the Prompter seam. Raises `AbortedByUser` when the
    user cancels any prompt, `BuildError` when the templates/models library
    is in an inconsistent state.
    """
    task_map = discover_tasks(template_dir)
    if not task_map:
        raise BuildError(f"No templates found under '{template_dir}'.")

    project_name = _prompt_for_project_name(initial_name, prompter)
    task = _prompt_for_task(task_map, prompter)
    method = _prompt_for_method(task, task_map, prompter, console)
    available_models = _discover_or_fail(task, method, models_package_path, console)
    selected_model = _prompt_for_model(available_models, prompter)
    description = _prompt_for_description(prompter)

    return BuildRequest(
        name=project_name,
        task=task,
        method=method,
        available_models=available_models,
        selected_model=selected_model,
        description=description,
    )


def _prompt_for_project_name(initial: str | None, prompter: Prompter) -> str:
    if initial:
        return initial

    answer = prompter.text("Enter the name of the build:")
    if not answer or not answer.strip():
        raise AbortedByUser("Project name is required.")
    return answer.strip()


def _prompt_for_task(task_map: dict[str, list[str]], prompter: Prompter) -> str:
    selected = prompter.select("Select the AI task:", list(task_map.keys()))
    if not selected:
        raise AbortedByUser("Task selection cancelled.")
    return selected


def _prompt_for_method(
    task: str,
    task_map: dict[str, list[str]],
    prompter: Prompter,
    console: Console,
) -> str:
    methods = [m for m in task_map.get(task, []) if m != "common"]
    if not methods:
        raise BuildError(f"No methods/templates found for the task '{task}'.")

    selected = prompter.select("Select the method:", methods)
    if not selected or selected not in methods:
        raise BuildError(f"Method '{selected}' does not exist for task '{task}'.")
    return selected


def _discover_or_fail(
    task: str, method: str, models_package_path: Path, console: Console
) -> list[str]:
    models = discover_model_names(task, method, models_package_path)
    if not models:
        raise BuildError("No models found for the selected task and method.")
    return models


def _prompt_for_model(choices: list[str], prompter: Prompter) -> str:
    selected = prompter.select("Select the default model for config.yaml:", choices)
    if not selected:
        raise AbortedByUser("Model selection cancelled.")
    return selected


def _prompt_for_description(prompter: Prompter) -> str:
    answer = prompter.text(
        "Enter the description of the build:",
        default="A GeoSave Engine project.",
    )
    if answer is None:
        raise AbortedByUser("Description input cancelled.")
    return answer
