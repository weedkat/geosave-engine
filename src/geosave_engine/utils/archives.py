from __future__ import annotations

import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, extract_to: Path, *, skip_if_extracted: bool = True) -> None:
    """Extract `zip_path` into `extract_to`, creating the directory if needed."""
    
    if skip_if_extracted and extract_to.exists() and any(extract_to.iterdir()):
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")


    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as handle:
        handle.extractall(extract_to)


def cleanup_zip(zip_path: Path) -> None:
    """Delete `zip_path` if it exists; no-op otherwise."""
    if zip_path.exists():
        zip_path.unlink()
