from pathlib import Path

FILE_EXCEPTION = ['__pycache__', 'common', '.ipynb_checkpoints']


def _package_dir() -> Path:
    return Path(__file__).parent.parent


def templates_dir() -> Path:
    return _package_dir() / "templates"


def common_template_dir() -> Path:
    return Path(__file__).parent / "common"


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
            "pixelwise_regression": {"ibm_granite_biomass": Path(...)},
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


def get_catalog_options(task: str, method: str, file_exceptions: list[str] = FILE_EXCEPTION) -> list[str]:
    """List available catalog names for a given task + method template.

    A catalog is a subdir of the method dir that contains its own ``modules/``.
    If the method dir itself has ``modules/``, it is a direct-copy template with no catalog.

    Returns:
        Sorted list of catalog directory names. Empty list if method is direct-copy.
    """
    method_dir = templates_dir() / task / method
    if not method_dir.exists():
        return []
    return sorted(
        p.name for p in method_dir.iterdir()
        if p.is_dir() and p.name not in file_exceptions and (p / "modules").exists()
    )


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
