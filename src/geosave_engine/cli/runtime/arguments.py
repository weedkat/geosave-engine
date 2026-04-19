from __future__ import annotations

from geosave_engine.cli.errors import AbortedByUser, WorkspaceError
from geosave_engine.cli.io import Prompter
from geosave_engine.cli.search import (
    ProjectLayout,
    find_artifact_parents,
    find_configs,
    list_user_scripts,
    resolve_user_script,
)
from geosave_engine.utils.strings import parse_shell_args


def resolve_config_args(
    config: str | None,
    layout: ProjectLayout,
    prompter: Prompter,
) -> list[str]:
    """Return the `--config <path>` pair for a Lightning CLI invocation."""
    if config:
        return ["--config", config]

    configs = find_configs(layout)
    if not configs:
        raise WorkspaceError(
            f"No .yaml/.yml configuration files found in '{layout.configs_dir}'."
        )

    selected = prompter.select_mapping(
        "Select the configuration file:",
        [(path.name, str(path.resolve())) for path in configs],
    )
    if not selected:
        raise AbortedByUser("Config selection cancelled.")
    return ["--config", selected]


def resolve_artifact_args(
    artifacts: str | None,
    layout: ProjectLayout,
    prompter: Prompter,
) -> list[str]:
    """Return the `--model <path>` pair pointing at a trained-artifact directory."""
    if artifacts:
        return ["--model", artifacts]

    parents = find_artifact_parents(layout)
    if not parents:
        raise WorkspaceError(
            "No valid artifact directories containing config files found in "
            f"'{layout.artifacts_dir}'."
        )

    selected = prompter.select_mapping(
        "Select the model artifacts containing config:",
        [(path.name, str(path.resolve())) for path in parents],
    )
    if not selected:
        raise AbortedByUser("Artifact selection cancelled.")
    return ["--model", selected]


def resolve_script_invocation(
    script_name: str | None,
    extra_args: list[str],
    layout: ProjectLayout,
    prompter: Prompter,
) -> tuple[str, list[str]]:
    """Determine which script under `scripts/` to run, plus remaining argv."""
    if script_name:
        return script_name, extra_args

    if extra_args and not extra_args[0].startswith("-"):
        return extra_args[0], extra_args[1:]

    scripts = list_user_scripts(layout)
    if not scripts:
        raise WorkspaceError(f"No scripts found in {layout.scripts_dir}.")

    choices = [
        (str(path.relative_to(layout.scripts_dir)), str(path.relative_to(layout.scripts_dir)))
        for path in scripts
    ]
    forwarded = f" (extra args {extra_args!r} will be forwarded)" if extra_args else ""
    selected = prompter.select_mapping(f"Select the script to run{forwarded}:", choices)
    if not selected:
        raise AbortedByUser("Script selection cancelled.")
    return selected, extra_args


def resolve_script_extra_args(extra_args: list[str], prompter: Prompter) -> list[str]:
    """Prompt the user for additional args if none were passed on the command line."""
    if extra_args:
        return extra_args

    answer = prompter.text("Enter additional script arguments (optional):", default="")
    if answer is None:
        raise AbortedByUser("Script argument input cancelled.")
    try:
        return parse_shell_args(answer)
    except ValueError as error:
        raise AbortedByUser(f"Invalid argument string: {error}") from error


def locate_user_script(layout: ProjectLayout, script_name: str) -> str:
    """Wrapper around `resolve_user_script` that raises `WorkspaceError` on miss."""
    path = resolve_user_script(layout, script_name)
    if path is None:
        raise WorkspaceError(
            f"Script '{script_name}' not found in {layout.scripts_dir}. "
            "Only .py scripts directly under scripts/ are supported."
        )
    return str(path)
