"""Filesystem resolution and location arithmetic for geodata I/O."""

from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import TypeGuard

import fsspec
from fsspec.core import split_protocol
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

type StorageOptions = Mapping[str, object]


def absolute_location(location: str | PathLike[str]) -> str:
    """Resolve local source paths while retaining remote URIs."""
    protocol, path = split_protocol(str(location))
    return (
        str(Path(path).resolve())
        if protocol in (None, "file", "local")
        else str(location)
    )


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


def resolve_asset_path(
    reference: str | PathLike[str],
    *,
    filesystem: AbstractFileSystem,
    catalog_path: str,
) -> Path | str:
    """Expand one relative asset pointer against a catalog parent."""
    spelled = str(reference)
    protocol, _ = split_protocol(spelled)
    if protocol is not None:
        return spelled

    local = Path(spelled)
    if local.is_absolute():
        return local

    if is_local_filesystem(filesystem):
        joined = os.path.abspath(Path(catalog_path).parent / local)
        return Path(joined)

    joined = posixpath.normpath(
        posixpath.join(posixpath.dirname(catalog_path), spelled)
    )
    return filesystem.unstrip_protocol(joined)


def stored_asset_path(
    reference: str | PathLike[str],
    *,
    filesystem: AbstractFileSystem,
    catalog_path: str,
) -> str:
    """Shorten an asset pointer only when it is below the catalog parent."""
    spelled = str(reference)
    protocol, protocol_path = split_protocol(spelled)
    protocols = filesystem.protocol
    filesystem_protocols = (
        (protocols,) if isinstance(protocols, str) else tuple(protocols)
    )

    if protocol is not None:
        if protocol not in filesystem_protocols:
            return spelled
        remote_parent = PurePosixPath(posixpath.dirname(catalog_path).lstrip("/"))
        remote_candidate = PurePosixPath(posixpath.normpath(protocol_path).lstrip("/"))
        try:
            return remote_candidate.relative_to(remote_parent).as_posix()
        except ValueError:
            return spelled

    local_candidate = Path(spelled)
    if not local_candidate.is_absolute():
        return local_candidate.as_posix()
    if not is_local_filesystem(filesystem):
        return spelled

    local_parent = Path(catalog_path).parent
    normalized = Path(os.path.abspath(local_candidate))
    try:
        return normalized.relative_to(local_parent).as_posix()
    except ValueError:
        return spelled
