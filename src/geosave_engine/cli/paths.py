from pathlib import Path
from typing import Annotated, Literal

METHOD_DIR = 'methods'
PLUGIN_DIR = 'plugins'
FILE_EXCEPTION = ['__pycache__', 'common']

template_struct = Annotated[
    dict[str, dict[str, Path]], 
    "Mapping of task names to lists of template paths."
]

plugins = Literal["scripts", "notebooks"]

def src_dir() -> Path:
    return Path(__file__).parent.parent.parent

def templates_dir() -> Path:
    return src_dir() / "templates"

def common_template_dir() -> Path:
    return templates_dir() / "common"


def get_task_templates(file_exceptions: list[str] = FILE_EXCEPTION) -> list[Path]:
    templates = [p for p in templates_dir().iterdir() if p.name not in file_exceptions and p.is_dir()]
    return templates

def get_method_templates(file_exceptions: list[str] = FILE_EXCEPTION) -> template_struct:
    """Get mapping of task names to lists of method template paths,

    Args: 
        file_exceptions: List of file or directory names to exclude from templates. Only top-level names are checked, not recursive. For example, if "common" is in file_exceptions, any template path containing a directory named "common" will be excluded, regardless of its position in the directory structure. This allows for flexible exclusion of templates based on their names,
    
    Returns:
        {
            "semantic_segmentation": [Path(...), Path(...)],
            "object_detection": [Path(...), Path(...)],
            ...
        }
    """
    tasks = get_task_templates(file_exceptions=file_exceptions)
    method_templates = {}
    for task in tasks:
        method_dir = task / METHOD_DIR
        if not method_dir.exists() or not method_dir.is_dir():
            continue
        task_templates = {p.name: p for p in method_dir.iterdir() if p.name not in file_exceptions and p.is_dir()}
        method_templates[task.name] = task_templates
    return method_templates

def get_plugin_templates(plugin: plugins, file_exceptions: list[str] = FILE_EXCEPTION) -> template_struct:
    """Get mapping of task names to lists of plugin template paths.
    
    Args:
        plugin: The plugin for which to retrieve templates.
        file_exceptions: List of file or directory names to exclude from templates. Only top-level
    
    Returns:
        {
            "semantic_segmentation": {"ingest": Path(...)},
            "object_detection": {},
            ...
        }
    """
    tasks = get_task_templates(file_exceptions=file_exceptions)
    plugin_templates = {}
    for task in tasks:
        plugin_dir = task / PLUGIN_DIR / plugin
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            continue
        task_templates = {p.name: p for p in plugin_dir.iterdir() if p.name not in file_exceptions}
        plugin_templates[task.name] = task_templates
    return plugin_templates


def get_workspace_scripts(scripts_dir: Path, file_exceptions: list[str] = FILE_EXCEPTION) -> dict[str, Path]:
    """Get runnable Python scripts from a workspace scripts directory.

    Returns a dictionary keyed by scripts-relative path
    (e.g. "dynamic_world_ingest/ingest.py") with absolute paths as values.
    """
    if not scripts_dir.exists() or not scripts_dir.is_dir():
        return {}

    scripts: dict[str, Path] = {}
    for path in sorted(scripts_dir.rglob("*.py")):
        if any(part in file_exceptions for part in path.parts):
            continue
        script_name = path.relative_to(scripts_dir).as_posix()
        scripts[script_name] = path.resolve()
    return scripts


def get_workspace_artifacts(artifacts_dir: Path) -> dict[str, Path]:
    """Discover Lightning-style artifacts with config.yaml files.

    Returns dict mapping 'model_name/version_0' -> path_to_config.yaml.
    """
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return {}

    artifacts = {}
    for model_dir in sorted(artifacts_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            config_path = version_dir / "config.yaml"
            if config_path.exists():
                key = f"{model_dir.name}/{version_dir.name}"
                artifacts[key] = config_path.resolve()
    return artifacts

if __name__ == "__main__":
    # print(get_task_templates())
    print(get_plugin_templates("scripts"))