import questionary
from geosave_engine.cli.paths import plugins, get_plugin_templates
from typing import get_args
from geosave_engine.cli.errors import AbortedByUserError

def prompt_for_plugin_type():
    answer = questionary.select(
        "Select the type of component to add:",
        choices=list(get_args(plugins))
    ).ask()
    if answer is None:
        raise AbortedByUserError("Plugin type selection was aborted by the user.")
    return answer.strip()

def prompt_for_plugin_name(plugin_type: plugins, task: str):
    plugin_templates = get_plugin_templates(plugin_type)[task]
    answer = questionary.select(
        "Select the component to add:",
        choices=list(plugin_templates.keys())
    ).ask()
    if answer is None:
        raise AbortedByUserError("Plugin name selection was aborted by the user.")
    return answer.strip()


def prompt_for_script_name(script_names: list[str]) -> str:
    answer = questionary.select(
        "Select a script to run:",
        choices=script_names,
    ).ask()
    if answer is None:
        raise AbortedByUserError("Script selection was aborted by the user.")
    return answer.strip()


def prompt_for_artifact(artifact_keys: list[str]) -> str:
    """Prompt user to select an artifact (model_name/version_N)."""
    answer = questionary.select(
        "Select an artifact:",
        choices=artifact_keys,
    ).ask()
    if answer is None:
        raise AbortedByUserError("Artifact selection was aborted by the user.")
    return answer.strip()


def prompt_for_config(config_paths: list[str]) -> str:
    """Prompt user to select a config file."""
    answer = questionary.select(
        "Select a config file:",
        choices=config_paths,
    ).ask()
    if answer is None:
        raise AbortedByUserError("Config selection was aborted by the user.")
    return answer.strip()