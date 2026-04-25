from __future__ import annotations

from pathlib import Path

from geosave_engine.cli.errors import WorkspaceError
from geosave_engine.cli.io import Console, Prompter
from geosave_engine.cli.runtime.arguments import (
    locate_user_script,
    resolve_artifact_args,
    resolve_config_args,
    resolve_script_extra_args,
    resolve_script_invocation,
)
from geosave_engine.cli.runtime.environment import load_environment
from geosave_engine.cli.runtime.executor import run_python_script
from geosave_engine.cli.runtime.workspace import (
    Workspace,
    announce_workspace,
    load_workspace,
)
from geosave_engine.cli.search import ProjectLayout, find_entrypoint


class ProjectScriptRunner:
    """Stateful coordinator for `fit` / `test` / `predict` / `run`.

    Holds the parsed workspace and the Prompter/Console seam so individual
    commands stay thin.
    """

    def __init__(
        self,
        project_dir: Path,
        current_dir: Path,
        *,
        prompter: Prompter,
        console: Console,
    ) -> None:
        self.project_dir = project_dir
        self.current_dir = current_dir
        self.prompter = prompter
        self.console = console

        self.workspace: Workspace = load_workspace(project_dir)
        announce_workspace(self.workspace, console)
        self.layout = ProjectLayout(root=project_dir)

    def fit(self, config: str | None, extra_args: list[str]) -> None:
        run_args = ["fit", *resolve_config_args(config, self.layout, self.prompter), *extra_args]
        self._run_entrypoint(run_args, operation="training")

    def test(
        self,
        artifacts: str | None,
        extra_args: list[str],
        config: str | None = None,
    ) -> None:
        run_args = [
            "test",
            *resolve_artifact_args(
                artifacts, self.layout, self.prompter, config_override=config
            ),
            *extra_args,
        ]
        self._run_entrypoint(run_args, operation="test")

    def predict(
        self,
        artifacts: str | None,
        extra_args: list[str],
        config: str | None = None,
    ) -> None:
        run_args = [
            "predict",
            *resolve_artifact_args(
                artifacts, self.layout, self.prompter, config_override=config
            ),
            *extra_args,
        ]
        self._run_entrypoint(run_args, operation="inference")

    def run(self, script_name: str | None, extra_args: list[str]) -> None:
        selected, remaining = resolve_script_invocation(
            script_name, extra_args, self.layout, self.prompter
        )
        run_args = resolve_script_extra_args(remaining, self.prompter)
        script_path = Path(locate_user_script(self.layout, selected))
        self._execute(script_path, run_args, operation=f"script '{selected}'")

    def _run_entrypoint(self, run_args: list[str], *, operation: str) -> None:
        entrypoint = find_entrypoint(self.layout, "main.py")
        if entrypoint is None:
            raise WorkspaceError(f"main.py not found in {self.project_dir}.")
        self._execute(entrypoint, run_args, operation=operation)

    def _execute(self, script_path: Path, run_args: list[str], *, operation: str) -> None:
        env_state = load_environment(self.project_dir, self.current_dir, self.console)

        args_str = " ".join(run_args) if run_args else "No extra args"
        script_rel = script_path.relative_to(self.project_dir)
        task_display = f" (Task: {self.workspace.task})" if self.workspace.task else ""
        self.console.success(
            f"Starting {operation} for '{self.workspace.project_name}'{task_display}\n"
            f"Executing: `python {script_rel} {args_str}`"
        )

        run_python_script(
            script_path,
            run_args,
            cwd=self.project_dir,
            env=env_state.env,
        )
