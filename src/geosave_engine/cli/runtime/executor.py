from __future__ import annotations

import subprocess
from pathlib import Path

from geosave_engine.cli.errors import ScriptError


def run_python_script(
    script_path: Path,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    """Invoke `python script_path *args` in `cwd` with the given `env`.

    Raises `ScriptError` (carrying the subprocess exit code) on failure.
    """
    try:
        subprocess.run(
            ["python", str(script_path.resolve()), *args],
            check=True,
            cwd=cwd,
            env=env,
        )
    except subprocess.CalledProcessError as error:
        raise ScriptError(
            f"Script {script_path.name} exited with code {error.returncode}.",
            exit_code=error.returncode,
        ) from error
