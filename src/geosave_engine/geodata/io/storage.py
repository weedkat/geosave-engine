"""Filesystem resolution and location arithmetic for geodata I/O."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from os import PathLike
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TypeGuard

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


def is_local_filesystem(
    filesystem: AbstractFileSystem,
) -> TypeGuard[LocalFileSystem]:
    """Return whether a filesystem uses native local paths."""
    return isinstance(filesystem, LocalFileSystem)


def gdal_path(location: str | PathLike[str]) -> str:
    """Spell a location the way GDAL opens it.

    GDAL reads local paths and `s3://`, `gs://`, `az://` and `https://` URLs
    itself. A Hugging Face bucket is reached through its S3 gateway, which
    needs S3 keys and `configure_gdal` pointed at `s3.hf.co` beforehand.

    Args:
        location: Local path or URL of a raster file.

    Returns:
        `s3://<namespace>/<bucket>/<key>` for `hf://buckets/<namespace>/<bucket>/<key>`,
        else the location unchanged.

    Raises:
        ValueError: An `hf://` URL names something other than a bucket.

    Examples:
        >>> gdal_path("hf://buckets/me/samples/forest/a.tif")
        's3://me/samples/forest/a.tif'
    """
    protocol, path = split_protocol(str(location))
    if protocol != "hf":
        return str(location)
    if not path.startswith("buckets/"):
        raise ValueError(
            f"{location} is not a Hugging Face bucket; only hf://buckets/ URLs "
            f"are served by the S3 gateway GDAL reads through"
        )
    return f"s3://{path.removeprefix('buckets/')}"


@contextmanager
def local_target(
    location: str | PathLike[str],
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> Iterator[Path]:
    """Yield a local path to write, then publish it at `location`.

    Args:
        location: Local path, or fsspec URL of the file to produce.
        overwrite: Replace an existing file or object.
        storage_options: Options for the location's filesystem.

    Yields:
        `location` itself when it is local, else a path of the same name in a
        temporary folder, uploaded once the block ends without an error.

    Raises:
        FileExistsError: `location` exists and `overwrite` is false.

    Examples:
        >>> with local_target("s3://bucket/scene.tif") as target:
        ...     write(target)
    """
    filesystem, path = filesystem_path(location, storage_options)
    if not overwrite and filesystem.exists(path):
        raise FileExistsError(f"{location} exists; pass overwrite=True to replace it")
    if is_local_filesystem(filesystem):
        yield Path(path)
        return
    with TemporaryDirectory() as folder:
        target = Path(folder) / PurePosixPath(path).name
        yield target
        filesystem.put(str(target), path)
