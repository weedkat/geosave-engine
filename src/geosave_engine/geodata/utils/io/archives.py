from __future__ import annotations

import zipfile
from pathlib import Path


def _cleanup_macosx_artifacts(root: Path) -> None:
    """Remove macOS archive metadata files extracted from a ZIP archive."""
    for path in root.rglob("__MACOSX"):
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
    for path in root.rglob("._*"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def extract_zip(zip_path: Path, extract_to: Path, *, skip_if_extracted: bool = True) -> None:
    """Extract `zip_path` into `extract_to`, creating the directory if needed."""
    
    if skip_if_extracted and extract_to.exists() and any(extract_to.iterdir()):
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")


    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as handle:
        handle.extractall(extract_to)
    _cleanup_macosx_artifacts(extract_to)


def cleanup_zip(zip_path: Path) -> None:
    """Delete `zip_path` if it exists; no-op otherwise."""
    if zip_path.exists():
        zip_path.unlink()
