import shutil
import questionary
from pathlib import Path

def safe_copy(src: Path | str, dst: Path | str, overwrite_all: bool = False, skip_all: bool = False) -> None:
    # Initialize state for the first call
    overwrite_all = overwrite_all
    skip_all = skip_all

    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source '{src}' does not exist.")

    # 1. Handle Files
    if src.is_file():
        if dst.exists() and not overwrite_all and not skip_all:
            choice = _ask_overwrite(dst)
            match choice:
                case "always":
                    overwrite_all = True
                case "skip_all":
                    skip_all = True
                case "skip":
                    return
                case "overwrite":
                    pass
        
        if not (dst.exists() and skip_all):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    elif src.is_dir():
        for item in src.iterdir():
            safe_copy(item, dst / item.name, overwrite_all=overwrite_all, skip_all=skip_all)

def _ask_overwrite(path: Path) -> str:
    return questionary.select(
        f"'{path.name}' already exists. What should I do?",
        choices=[
            {"name": "Overwrite", "value": "overwrite"},
            {"name": "Skip", "value": "skip"},
            {"name": "Overwrite All", "value": "always"},
            {"name": "Skip All Remaining", "value": "skip_all"},
        ]
    ).ask()