from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Iterable


def _matches_pattern(rel_path: str, name: str, patterns: Iterable[str]) -> bool:
    prefixed = f"./{rel_path}"
    return any(
        fnmatch.fnmatch(rel_path, pattern)
        or fnmatch.fnmatch(prefixed, pattern)
        or fnmatch.fnmatch(name, pattern)
        for pattern in patterns
    )


def _build_ignore_callback(source: Path, patterns: tuple[str, ...]):
    def ignore_patterns(current_dir: str, names: list[str]) -> set[str]:
        current_path = Path(current_dir)
        ignored: set[str] = set()
        for name in names:
            candidate = current_path / name
            rel_path = candidate.relative_to(source).as_posix()
            if _matches_pattern(rel_path, name, patterns):
                ignored.add(name)
        return ignored

    return ignore_patterns


def copy_tree(
    source: Path,
    destination: Path,
    *,
    file_exceptions: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Recursively copy `source` to `destination`, skipping files that match any pattern.

    Overwrites existing files in-place (`dirs_exist_ok=True`). The caller is
    responsible for deciding whether to proceed when `destination` exists.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    patterns = tuple(file_exceptions or ())

    if source.is_file():
        if patterns and _matches_pattern(source.name, source.name, patterns):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return

    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=_build_ignore_callback(source, patterns) if patterns else None,
    )
