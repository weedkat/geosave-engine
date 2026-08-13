import fnmatch
from pathlib import Path
import shutil
import questionary
from typing import Iterable


class CopyState:
    """Holds mutable state across recursive calls."""
    def __init__(self, overwrite_all: bool = False, skip_all: bool = False):
        self.overwrite_all = overwrite_all
        self.skip_all = skip_all


def safe_copy(
    src: Path | str,
    dst: Path | str,
    overwrite_all: bool = False,
    skip_all: bool = False,
    exclude: Iterable[str] | None = None,
) -> None:
    """Recursively copies files/directories with interactive collision handling and exclusions.

    Args:
        src: Source path.
        dst: Destination path.
        overwrite_all: Pre-set flag to overwrite all existing files without asking.
        skip_all: Pre-set flag to skip all existing files without asking.
        exclude: List/set of glob patterns to skip (e.g., ["*.pyc", "__pycache__", ".git"]).
    """
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Source '{src_path}' does not exist.")

    state = CopyState(overwrite_all=overwrite_all, skip_all=skip_all)
    patterns = list(exclude) if exclude else []

    _copy_recursive(src_path, dst_path, state, patterns)


def _is_excluded(path: Path, patterns: Iterable[str]) -> bool:
    """Checks if a file or directory matches any exclude pattern."""
    for pattern in patterns:
        if fnmatch.fnmatch(path.name, pattern) or path.match(pattern):
            return True
    return False


def _copy_recursive(
    src: Path,
    dst: Path,
    state: CopyState,
    exclude: Iterable[str],
) -> None:
    if _is_excluded(src, exclude):
        return

    # Handle Files
    if src.is_file():
        if dst.exists():
            if state.skip_all:
                return

            if not state.overwrite_all:
                choice = _ask_overwrite(dst)
                match choice:
                    case "always":
                        state.overwrite_all = True
                    case "skip_all":
                        state.skip_all = True
                        return
                    case "skip":
                        return
                    case "overwrite":
                        pass
                    case None:  # User aborted with Ctrl+C / Esc
                        raise KeyboardInterrupt("Copy process cancelled by user.")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Handle Directories
    elif src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            _copy_recursive(item, dst / item.name, state, exclude)


def _ask_overwrite(path: Path) -> str | None:
    return questionary.select(
        f"'{path.name}' already exists. What should I do?",
        choices=[
            questionary.Choice("Overwrite", value="overwrite"),
            questionary.Choice("Skip", value="skip"),
            questionary.Choice("Overwrite All", value="always"),
            questionary.Choice("Skip All Remaining", value="skip_all"),
        ],
    ).ask()