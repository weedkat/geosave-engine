from pathlib import Path

FILE_EXCEPTION = ['__pycache__', 'common', '.ipynb_checkpoints']


def _package_dir() -> Path:
    return Path(__file__).parent.parent

def templates_dir() -> Path:
    return _package_dir() / "templates"

def common_template_dir() -> Path:
    return Path(__file__).parent / "common"

def plugins_dir() -> Path:
    return _package_dir() / "plugins"


def get_task_templates(file_exceptions: list[str] = FILE_EXCEPTION) -> list[Path]:
    templates = [p for p in templates_dir().iterdir() if p.name not in file_exceptions and p.is_dir()]
    return templates

def get_method_templates(file_exceptions: list[str] = FILE_EXCEPTION) -> dict[str, dict[str, Path]]:
    """Get mapping of task names to method template paths.

    Args:
        file_exceptions: Directory names to exclude from discovery.

    Returns:
        {
            "semantic_segmentation": {"supervised_dw": Path(...)},
            "object_detection": {"supervised": Path(...)},
            ...
        }
    """
    tasks = get_task_templates(file_exceptions=file_exceptions)
    method_templates = {}
    for task in tasks:
        task_templates = {
            p.name: p
            for p in task.iterdir()
            if p.is_dir() and p.name not in file_exceptions
        }
        if task_templates:
            method_templates[task.name] = task_templates
    return method_templates


def get_plugin_templates(file_exceptions: list[str] = FILE_EXCEPTION) -> dict[str, Path]:
    """Get all available plugin template paths keyed by namespaced path.

    Args:
        file_exceptions: Directory names to exclude.

    Returns:
        {
            "scripts/dynamicworld": Path(...),
            "notebooks/tutorial": Path(...),
            ...
        }
    """
    pd = plugins_dir()
    if not pd.exists() or not pd.is_dir():
        return {}
    result: dict[str, Path] = {}
    for type_dir in sorted(pd.iterdir()):
        if not type_dir.is_dir() or type_dir.name in file_exceptions:
            continue
        for plugin in sorted(type_dir.iterdir()):
            if plugin.is_dir() and plugin.name not in file_exceptions:
                result[f"{type_dir.name}/{plugin.name}"] = plugin
    return result


def get_workspace_scripts(scripts_dir: Path, file_exceptions: list[str] = FILE_EXCEPTION) -> dict[str, Path]:
    """Get runnable Python scripts from a workspace scripts directory.

    Returns:
        {"dynamic_world_ingest/ingest.py": Path(...), ...}
    """
    if not scripts_dir.exists() or not scripts_dir.is_dir():
        return {}
    scripts: dict[str, Path] = {}
    for path in sorted(scripts_dir.rglob("*.py")):
        if any(part in file_exceptions for part in path.parts):
            continue
        scripts[path.relative_to(scripts_dir).as_posix()] = path.resolve()
    return scripts


def get_workspace_notebooks(notebooks_dir: Path, file_exceptions: list[str] = FILE_EXCEPTION) -> dict[str, Path]:
    """Get Jupyter notebooks from a workspace notebooks directory.

    Returns:
        {"tutorial.ipynb": Path(...), ...}
    """
    if not notebooks_dir.exists() or not notebooks_dir.is_dir():
        return {}
    notebooks: dict[str, Path] = {}
    for path in sorted(notebooks_dir.rglob("*.ipynb")):
        if any(part in file_exceptions for part in path.parts):
            continue
        notebooks[path.relative_to(notebooks_dir).as_posix()] = path.resolve()
    return notebooks


def get_workspace_artifacts(artifacts_dir: Path) -> dict[str, Path]:
    """Discover Lightning-style artifacts with config.yaml files.

    Returns:
        {"model_name/version_0": Path(...), ...}
    """
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return {}
    artifacts: dict[str, Path] = {}
    for model_dir in sorted(artifacts_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            config_path = version_dir / "config.yaml"
            if config_path.exists():
                artifacts[f"{model_dir.name}/{version_dir.name}"] = config_path.resolve()
    return artifacts
