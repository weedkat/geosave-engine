from __future__ import annotations

import ast
from pathlib import Path

import typer


class ProjectSearch:
    def __init__(self, project_dir: Path | None = None) -> None:
        self.project_dir = project_dir

    @staticmethod
    def get_tasks(template_path: Path) -> dict[str, list[str]]:
        tasks: dict[str, list[str]] = {}
        folders = [
            folder
            for folder in template_path.iterdir()
            if folder.is_dir() and folder.name not in {"__pycache__", "common"}
        ]
        for folder in folders:
            methods = [
                method
                for method in folder.iterdir()
                if method.is_dir() and method.name != "__pycache__"
            ]
            task_name = folder.name.replace("_", " ").lower()
            tasks[task_name] = [method.name.replace("_", " ").lower() for method in methods]
        return tasks

    @staticmethod
    def get_model_list(task: str, method: str, package_path: Path) -> list[str]:
        class_names: list[str] = []
        for file in package_path.glob("**/build.py"):
            if file.name == "__init__.py":
                continue

            with open(file, "r", encoding="utf-8") as handle:
                node = ast.parse(handle.read())

            for class_node in node.body:
                if not isinstance(class_node, ast.ClassDef):
                    continue

                for item in class_node.body:
                    if not isinstance(item, ast.Assign):
                        continue

                    for target in item.targets:
                        if not (isinstance(target, ast.Name) and target.id == "tasks"):
                            continue
                        if not isinstance(item.value, ast.Dict):
                            continue

                        for key, value in zip(item.value.keys, item.value.values):
                            if not (isinstance(key, ast.Constant) and key.value == task):
                                continue
                            if not isinstance(value, ast.List):
                                continue

                            methods = [
                                elt.value
                                for elt in value.elts
                                if isinstance(elt, ast.Constant)
                            ]
                            if not method or len(methods) == 0 or method in methods:
                                class_names.append(class_node.name)

        return class_names

    def _require_project_dir(self) -> Path:
        if self.project_dir is None:
            raise ValueError("ProjectSearch requires project_dir for runtime search operations.")
        return self.project_dir

    def get_project_dir(self) -> Path:
        return self._require_project_dir()

    def find_configs(self) -> list[Path]:
        project_dir = self._require_project_dir()
        configs_dir = project_dir / "configs"
        if configs_dir.exists() and configs_dir.is_dir():
            return list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
        return []

    def find_artifact_parents(self) -> list[Path]:
        project_dir = self._require_project_dir()
        artifacts_path = project_dir / "artifacts"
        choices: list[Path] = []
        if artifacts_path.exists() and artifacts_path.is_dir():
            for directory in artifacts_path.iterdir():
                if directory.is_dir() and (
                    list(directory.glob("*.yaml")) or list(directory.glob("*.yml"))
                ):
                    choices.append(directory)
        return choices

    def find_script(self, script_names: str | list[str]) -> Path:
        project_dir = self._require_project_dir()
        names = [script_names] if isinstance(script_names, str) else script_names

        for name in names:
            scripts = list(project_dir.rglob(name))
            if scripts:
                return scripts[0]

        typer.secho(
            f"Error: {names[0]} not found in {project_dir}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    def find_script_in_scripts_dir(self, script_name: str) -> Path:
        project_dir = self._require_project_dir()
        scripts_dir = project_dir / "scripts"

        if not scripts_dir.exists() or not scripts_dir.is_dir():
            typer.secho(
                f"Error: scripts directory not found in {project_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        script_name_path = Path(script_name)
        if script_name_path.suffix and script_name_path.suffix != ".py":
            typer.secho(
                f"Error: Unsupported script extension '{script_name_path.suffix}'. Only .py scripts are supported.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        candidates = [scripts_dir / f"{script_name}.py"] if script_name_path.suffix == "" else [scripts_dir / script_name]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        candidate_list = ", ".join(str(path.relative_to(project_dir)) for path in candidates)
        typer.secho(
            f"Error: Script '{script_name}' not found in scripts directory. Looked for: {candidate_list}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    def list_scripts_in_scripts_dir(self) -> list[Path]:
        project_dir = self._require_project_dir()
        scripts_dir = project_dir / "scripts"

        if not scripts_dir.exists() or not scripts_dir.is_dir():
            typer.secho(
                f"Error: scripts directory not found in {project_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        scripts = sorted(
            [
                path
                for path in scripts_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix == ".py"
                    and path.name != "__init__.py"
                    and "__pycache__" not in path.parts
                )
            ],
            key=lambda path: str(path.relative_to(scripts_dir)).lower(),
        )

        if not scripts:
            typer.secho(
                f"Error: No scripts found in {scripts_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        return scripts
