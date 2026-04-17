from __future__ import annotations

from pathlib import Path

from geosave_engine.cli.services.runtime.argument_selector import ArgumentSelector
from geosave_engine.cli.services.search.project_search import ProjectSearch
from geosave_engine.cli.services.runtime.workspace_validator import WorkspaceValidator
from geosave_engine.cli.services.runtime.project_script_service import ProjectScriptService


class ProjectScriptRunner:
    def __init__(self, project_dir: Path, current_dir: Path) -> None:
        self.project_dir = project_dir
        self.current_dir = current_dir
        self.workspace_config = WorkspaceValidator.validate_with_feedback(project_dir)
        self.project_search = ProjectSearch(project_dir)
        self.argument_selector = ArgumentSelector(self.project_search)

    def fit(self, config: str | None, extra_args: list[str]) -> None:
        run_args = ["fit"]
        run_args.extend(self.argument_selector.config_args(config))
        run_args.extend(extra_args)
        self._run(run_args, operation="training")

    def test(self, artifacts: str | None, extra_args: list[str]) -> None:
        run_args = ["test"]
        run_args.extend(self.argument_selector.artifact_args(artifacts))
        run_args.extend(extra_args)
        self._run(run_args, operation="test")

    def predict(self, artifacts: str | None, extra_args: list[str]) -> None:
        run_args = ["predict"]
        run_args.extend(self.argument_selector.artifact_args(artifacts))
        run_args.extend(extra_args)
        self._run(run_args, operation="inference")

    def run(self, script_name: str | None, extra_args: list[str]) -> None:
        selected_script, remaining_args = self.argument_selector.resolve_script_and_args(
            script_name,
            extra_args,
        )
        run_args = self.argument_selector.script_args(remaining_args)
        script_path = self.project_search.find_script_in_scripts_dir(selected_script)
        self._run(
            run_args,
            operation=f"script '{selected_script}'",
            script_path=script_path,
        )

    def _run(self, run_args: list[str], operation: str, script_path: Path | None = None) -> None:
        target_script = script_path or self.project_search.find_script("main.py")
        ProjectScriptService.execute(
            project_name=self.workspace_config.get("project_name", "Unknown"),
            task_name=self.workspace_config.get("task", ""),
            script_path=target_script,
            project_dir=self.project_dir,
            current_dir=self.current_dir,
            run_args=run_args,
            operation=operation,
        )
