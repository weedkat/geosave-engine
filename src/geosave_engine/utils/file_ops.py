from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Iterable

import questionary
import typer


def _matches_exception(path: str, name: str, patterns: Iterable[str]) -> bool:
    prefixed = f"./{path}"
    return any(
        fnmatch.fnmatch(path, pattern)
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
            if _matches_exception(rel_path, name, patterns):
                ignored.add(name)

        return ignored

    return ignore_patterns


def _copy_file(source: Path, destination: Path, patterns: tuple[str, ...]) -> bool:
    source_name = source.name
    if _matches_exception(source_name, source_name, patterns):
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_path(
    source: Path,
    destination: Path,
    file_exceptions: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_on_existing: bool = True,
) -> bool:
    exception_patterns = tuple(file_exceptions or ())

    if not source.exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    if destination.exists() and prompt_on_existing:
        overwrite = questionary.confirm(
            f"Warning: Destination '{destination}' already exists. Overwrite?"
        ).ask()
        if not overwrite:
            typer.secho("Copy operation cancelled.", fg=typer.colors.YELLOW)
            return False

    try:
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
                ignore=_build_ignore_callback(source, exception_patterns)
                if exception_patterns
                else None,
            )
        else:
            return _copy_file(source, destination, exception_patterns)
    except IOError as error:
        raise IOError(f"Error copying from '{source}' to '{destination}': {error}")

    return True
