"""Locations, filesystems, and the one way a single file is written."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import fsspec
from fsspec.core import split_protocol
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

type StorageOptions = Mapping[str, object]


def is_local(location: str | PathLike[str]) -> bool:
    """Return whether a location names a path on this machine."""
    protocol, _ = split_protocol(str(location))
    return protocol in (None, "file", "local")


def local_path(location: str | PathLike[str]) -> Path:
    """Read a local location as a path, without the `file://` it may carry."""
    _, path = split_protocol(str(location))
    return Path(path)


def absolute_location(location: str | PathLike[str]) -> str:
    """Resolve local source paths while retaining remote URIs."""
    if is_local(location):
        return str(local_path(location).resolve())
    return str(location)


def filesystem_path(
    location: str | PathLike[str],
    storage_options: StorageOptions | None = None,
) -> tuple[AbstractFileSystem, str]:
    """Resolve a location to its native filesystem and protocol-free path."""
    return fsspec.core.url_to_fs(str(location), **dict(storage_options or {}))


def write(
    path: str | PathLike[str],
    save: Callable[[Path], None],
    *,
    suffixes: Sequence[str],
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> Path | str:
    """Write one file through `save`, leaving its destination whole or untouched.

    Every single-file format writes this way: `save` only ever sees a local
    path, and the finished file is moved or uploaded to where it belongs.

    Args:
        path: Output path or fsspec URL.
        save: Writes the file at the local path it is handed.
        suffixes: Lowercase suffixes the destination may end in.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Local writes return a path; URL writes return the supplied URL.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong.

    Examples:
        >>> write("plots.parquet", plots.to_parquet, suffixes=(".parquet",))
        PosixPath('plots.parquet')
    """
    filesystem, target = filesystem_path(path, storage_options)
    name = PurePosixPath(target)
    if name.suffix.lower() not in suffixes:
        raise ValueError(
            f"destination {name.name!r} must end in one of {list(suffixes)}"
        )
    if not overwrite and filesystem.exists(target):
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    # A remote file is written in a temporary folder, then uploaded.
    if not isinstance(filesystem, LocalFileSystem):
        with TemporaryDirectory() as folder:
            staging = Path(folder) / name.name
            save(staging)
            filesystem.put(str(staging), target)
        return str(path)

    # A local file is written beside its destination, then moved over it.
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.stem}.staging{destination.suffix}")
    try:
        save(staging)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    return destination
