import re
from datetime import datetime
import getpass
from pathlib import Path
import platform
import tomlkit

def get_version() -> str:
    """Extract __version__ from __about__.py at Path(__file__).parents[2]."""
    about_path = Path(__file__).parents[2] / "__about__.py"
    
    if about_path.exists():
        content = about_path.read_text(encoding="utf-8")
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
            
    return "0.1.0"  # Fallback version if file/match is missing

def create_toml(
    target_dir: Path,
    name: str,
    task: str,
    method: str,
    description: str | None = None,
) -> Path:
    """Generates a lean project anchor TOML file."""
    toml_path = target_dir / "geosave.toml"
    doc = tomlkit.document()

    # --- [project] ---
    project = tomlkit.table()
    project.add("name", name)
    if description:
        project.add("description", description)
    project.add("version", get_version())
    project.add("created_at", datetime.now().astimezone())
    project.add("created_by", getpass.getuser())
    doc.add("project", project)

    # --- [workspace] ---
    workspace = tomlkit.table()
    workspace.add("task", task)
    workspace.add("method", method)
    doc.add("workspace", workspace)

    # --- [environment] ---
    env = tomlkit.table()
    env.add("geosave_version", get_version())
    env.add("python_version", platform.python_version())
    env.add("platform", platform.platform(terse=True))
    doc.add("environment", env)

    toml_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return toml_path