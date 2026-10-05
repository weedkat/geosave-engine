"""Original Dynamic World label releases from PANGAEA and Zenodo."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import BinaryIO, Literal, cast
from zipfile import ZipFile

import fsspec

_PANGAEA = "https://download.pangaea.de/dataset/933475/files"
_ARCHIVES = {
    "experts": (f"{_PANGAEA}/Experts_tiles.zip", None),
    "non_expert": (f"{_PANGAEA}/Non_expert_tiles.zip", None),
    "validation": (f"{_PANGAEA}/validation_set_tiles.zip", None),
    "test": (
        "https://zenodo.org/records/4766508/files/dw_test.zip?download=1",
        "9bff6f3fc346cf458ab21a5478f5a9fe",
    ),
}
_README = {"README.txt": f"{_PANGAEA}/README.txt"}
_TRAINING_METADATA = {
    **_README,
    "v1_dw_tile_metadata_for_public_release.xlsx": (
        f"{_PANGAEA}/v1_dw_tile_metadata_for_public_release.xlsx"
    ),
}
_METADATA = {
    "experts": _TRAINING_METADATA,
    "non_expert": _TRAINING_METADATA,
    "validation": _README,
    "test": {},
}
_COMPLETE = ".geosave-complete.json"


def _download_file(url: str, target: Path) -> None:
    """Stream one publisher file to a temporary local path."""
    with fsspec.open(url, "rb", block_size=0) as source, target.open("wb") as output:
        shutil.copyfileobj(cast(BinaryIO, source), output, length=1024 * 1024)


def download(
    root: str | Path,
    *,
    subset: Literal["experts", "non_expert", "validation", "test"] = "experts",
) -> Path:
    """Download and extract one original Dynamic World subset.

    Preserve the publisher's archive layout, labels, and metadata. Completed
    downloads are reused; temporary files are removed after a failed attempt.
    Imagery acquisition and conversion to a training catalog are separate steps.

    Args:
        root: Local directory in which to create the subset directory.
        subset: Expert or non-expert training labels, validation, or test files.

    Returns:
        Extracted directory at ``root / subset``, with accompanying metadata.

    Raises:
        ValueError: Unknown subset, checksum mismatch, unsafe archive path,
            or malformed completion JSON.
        FileExistsError: The destination exists without a matching completion
            record. Move or remove it before retrying.
        OSError: Transfer, extraction, or local filesystem failure.
        zipfile.BadZipFile: The downloaded archive is corrupt.

    Examples:
        >>> from geosave_engine.geodata.benchmarks import dynamic_world
        >>> labels = dynamic_world.download("data/dynamic_world", subset="experts")
        >>> labels
        PosixPath('data/dynamic_world/experts')
    """
    if subset not in _ARCHIVES:
        raise ValueError(
            f"Unknown Dynamic World subset {subset!r}; choose {tuple(_ARCHIVES)}"
        )
    url, checksum = _ARCHIVES[subset]
    metadata = _METADATA[subset]
    record = {"archive": url, "md5": checksum, "metadata": metadata}
    root = Path(root)
    destination = root / subset
    complete = destination / _COMPLETE
    if destination.exists():
        if complete.is_file() and json.loads(complete.read_text()) == record:
            return destination
        raise FileExistsError(
            f"{destination} is incomplete or belongs to another release; "
            "move or remove it before downloading"
        )

    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{subset}-", dir=root) as temporary:
        staging = Path(temporary)
        archive_path = staging / "archive.zip"
        _download_file(url, archive_path)
        if checksum is not None:
            with archive_path.open("rb") as archive_file:
                actual = hashlib.file_digest(archive_file, "md5").hexdigest()
            if actual != checksum:
                raise ValueError(f"Dynamic World {subset} checksum mismatch: {actual}")

        extracted = staging / "files"
        extracted.mkdir()
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename.replace("\\", "/"))
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or PureWindowsPath(member.filename).drive
                ):
                    raise ValueError(f"Unsafe archive member: {member.filename!r}")
            archive.extractall(extracted)
        for filename, source in metadata.items():
            _download_file(source, extracted / filename)
        (extracted / _COMPLETE).write_text(json.dumps(record, indent=2) + "\n")
        extracted.rename(destination)
    return destination
