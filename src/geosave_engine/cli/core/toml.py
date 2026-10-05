from datetime import datetime
import getpass
from pathlib import Path
import platform
import tomlkit

from geosave_engine.__about__ import __version__

_INITIAL_PROJECT_VERSION = "0.1.0"


def create_toml(
    target_dir: Path,
    name: str,
    workspace: str | None,
    description: str | None = None,
) -> Path:
    """Write the `geosave.toml` anchoring one generated workspace.

    Args:
        target_dir: Workspace root the file is written into.
        name: Project name.
        workspace: Workspace starter name, if any.
        description: Free-text project description.

    Returns:
        Path of the written file.
    """
    toml_path = target_dir / "geosave.toml"
    doc = tomlkit.document()

    project = tomlkit.table()
    project.add("name", name)
    if description:
        project.add("description", description)
    project.add("version", _INITIAL_PROJECT_VERSION)
    project.add("created_at", datetime.now().astimezone())
    project.add("created_by", getpass.getuser())
    doc.add("project", project)

    selection = tomlkit.table()
    if workspace is not None:
        selection.add("template", workspace)
    doc.add("workspace", selection)

    env = tomlkit.table()
    env.add("geosave_version", __version__)
    env.add("python_version", platform.python_version())
    env.add("platform", platform.platform(terse=True))
    doc.add("environment", env)

    toml_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return toml_path
