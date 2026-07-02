from pathlib import Path

import questionary

from geosave_engine.cli.errors import AbortedByUserError


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
