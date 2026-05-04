import questionary
from geosave_engine.cli.workspace import WorkspaceSpec
from geosave_engine.cli.paths import get_method_templates

method_templates = get_method_templates()

def create_scaffold():
    project_name = _ask_project_name()
    project_task = _ask_project_task()
    project_method = _ask_project_method(project_task)
    description = _ask_description() 

    return WorkspaceSpec(
        project_name=project_name,
        project_task=project_task,
        project_method=project_method,
        description=description
    )

def _ask_project_name() -> str:
    answer = questionary.text("Enter the project name:").ask()
    if not answer:
        raise ValueError("Project name is required.")
    return answer.strip()

def _ask_project_task() -> str:
    answer = questionary.select(
        "Select the main task for the project:",
        choices=list(method_templates.keys())
    ).ask()
    return answer.strip()

def _ask_project_method(task: str) -> str:
    answer = questionary.select(
        "Select the method for the project:",
        choices=list(str(m) for m in method_templates[task])
    ).ask()
    return answer.strip()

def _ask_description() -> str | None:
    answer = questionary.text("Enter a description for the project (optional):").ask()
    return answer.strip() if answer else None
