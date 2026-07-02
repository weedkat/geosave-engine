from pathlib import Path

import questionary

from geosave_engine.cli.errors import AbortedByUserError


def prompt_for_plugin(plugin_templates: dict[str, Path]) -> str:
    """Prompt user to select a plugin by namespaced path (e.g. 'scripts/dynamicworld').

    Args:
        plugin_templates: Mapping of namespaced key to source path.

    Returns:
        Selected namespaced key.

    Raises:
        AbortedByUserError: If user cancels.
    """
    answer = questionary.select(
        "Select a plugin to add:",
        choices=sorted(plugin_templates.keys()),
    ).ask()
    if answer is None:
        raise AbortedByUserError("Plugin selection was aborted by the user.")
    return answer.strip()


def prompt_for_runnable(runnables: list[str]) -> str:
    """Prompt user to select a script or notebook to run.

    Args:
        runnables: List of runnable keys (scripts and notebooks combined).

    Returns:
        Selected key.

    Raises:
        AbortedByUserError: If user cancels.
    """
    answer = questionary.select(
        "Select a script or notebook to run:",
        choices=runnables,
    ).ask()
    if answer is None:
        raise AbortedByUserError("Runnable selection was aborted by the user.")
    return answer.strip()


def prompt_for_artifact(artifact_keys: list[str]) -> str:
    """Prompt user to select an artifact (model_name/version_N).

    Raises:
        AbortedByUserError: If user cancels.
    """
    answer = questionary.select(
        "Select an artifact:",
        choices=artifact_keys,
    ).ask()
    if answer is None:
        raise AbortedByUserError("Artifact selection was aborted by the user.")
    return answer.strip()


def prompt_for_config(config_paths: list[str]) -> str:
    """Prompt user to select a config file.

    Raises:
        AbortedByUserError: If user cancels.
    """
    answer = questionary.select(
        "Select a config file:",
        choices=config_paths,
    ).ask()
    if answer is None:
        raise AbortedByUserError("Config selection was aborted by the user.")
    return answer.strip()
