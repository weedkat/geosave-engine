from __future__ import annotations

from pathlib import Path

from geosave_engine.cli.errors import AbortedByUserError, WorkspaceError
from geosave_engine.cli.io import Prompter
from geosave_engine.cli.search import (
    ProjectLayout,
    find_artifact_parents,
    find_configs,
    list_user_scripts,
    resolve_user_script,
)
from geosave_engine.utils.cli.strings import parse_shell_args


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
        raise AbortedByUserError("Config selection cancelled.")
    return ["--config", selected]


def resolve_artifact_args(
    artifacts: str | None,
    layout: ProjectLayout,
    prompter: Prompter,
    *,
    config_override: str | None = None,
) -> list[str]:
    """Resolve ``--artifacts <dir>`` to Lightning subcommand args.

    Expands the artifact directory into ``--config <dir>/config.yaml`` and the
    latest ``--ckpt_path <dir>/checkpoints/*.ckpt`` (when present). If
    ``config_override`` is provided, that ``--config`` is used instead of the
    one inside the artifact dir.
    """
    selected_dir: Path | None = None
    if artifacts:
        selected_dir = Path(artifacts).resolve()
    else:
        parents = find_artifact_parents(layout)
        if not parents:
            raise WorkspaceError(
                "No valid artifact directories containing config files found in "
                f"'{layout.artifacts_dir}'."
            )
        choices = [
            (
                str(path.relative_to(layout.artifacts_dir)),
                str(path.resolve()),
            )
            for path in parents
        ]
        selected = prompter.select_mapping(
            "Select the model artifacts containing config:",
            choices,
        )
        if not selected:
            raise AbortedByUserError("Artifact selection cancelled.")
        selected_dir = Path(selected)

    if not selected_dir.is_dir():
        raise WorkspaceError(f"Artifact directory not found: {selected_dir}")

    args: list[str] = []
    if config_override:
        args.extend(["--config", config_override])
    else:
        artifact_cfg = _artifact_config_path(selected_dir)
        if artifact_cfg is None:
            raise WorkspaceError(
                f"No config.yaml found inside artifact dir '{selected_dir}'."
            )
        args.extend(["--config", str(artifact_cfg)])

    ckpt = _latest_checkpoint(selected_dir)
    if ckpt is not None:
        args.extend(["--ckpt_path", str(ckpt)])
    return args


def _artifact_config_path(artifact_dir: Path) -> Path | None:
    for name in ("config.yaml", "config.yml"):
        candidate = artifact_dir / name
        if candidate.is_file():
            return candidate
    return None


def _latest_checkpoint(artifact_dir: Path) -> Path | None:
    ckpt_dir = artifact_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


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
        raise AbortedByUserError("Script selection cancelled.")
    return selected, extra_args


def resolve_script_extra_args(extra_args: list[str], prompter: Prompter) -> list[str]:
    """Prompt the user for additional args if none were passed on the command line."""
    if extra_args:
        return extra_args

    answer = prompter.text("Enter additional script arguments (optional):", default="")
    if answer is None:
        raise AbortedByUserError("Script argument input cancelled.")
    try:
        return parse_shell_args(answer)
    except ValueError as error:
        raise AbortedByUserError(f"Invalid argument string: {error}") from error


def locate_user_script(layout: ProjectLayout, script_name: str) -> str:
    """Wrapper around `resolve_user_script` that raises `WorkspaceError` on miss."""
    path = resolve_user_script(layout, script_name)
    if path is None:
        raise WorkspaceError(
            f"Script '{script_name}' not found in {layout.scripts_dir}. "
            "Only .py scripts directly under scripts/ are supported."
        )
    return str(path)
