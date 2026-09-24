# Remote GeoParquet Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read and write GeoParquet catalogs through local or fsspec-backed locations while materializing each catalog asset pointer into a directly usable local path or remote URL.

**Architecture:** A small `geodata.utils.io.storage` module owns filesystem resolution and lexical location arithmetic only. GeoParquet keeps ownership of serialization, local atomic replacement, remote overwrite behavior, and catalog path normalization; `read_vector` materializes relative asset pointers and returns a stateless `GeoVector`.

**Tech Stack:** Python 3.12, GeoPandas, PyArrow, fsspec, Hugging Face `HfFileSystem`, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-09-24-remote-persistence-design.md`

## Global Constraints

- Public persistence APIs remain format-oriented: `read_vector` and `GeoVector.to_geoparquet`.
- `storage_options` is exactly `Mapping[str, object] | None`; provider-specific top-level credential parameters are not added.
- Credentials remain transient and are never stored on `GeoVector`, written to catalog columns, or included in logs.
- The catalog `path` column is a pointer only; catalog persistence never verifies, uploads, or deletes the referenced asset.
- A relative pointer is relative to the catalog parent; an absolute local path or URL outside that parent remains absolute.
- Local GeoParquet writes keep staging plus atomic `os.replace`; remote replacement is not advertised as atomic.
- A failed remote write may clean up only a destination that did not exist before that call.
- Remote GeoTIFF, NetCDF, Zarr, and S3 support remain outside this implementation.
- Add fsspec as a direct runtime dependency; do not add a provider registry, filesystem wrapper class, or credential model.
- Preserve unrelated worktree changes. Before each commit, inspect every listed path; if it contains pre-existing changes outside this plan, skip that task commit and report it rather than committing another change accidentally.

## Review Focus

- An absolute URL on another provider, such as `s3://other-bucket/a.zarr`, must survive a catalog round trip without importing that provider's filesystem driver or contacting the asset.
- A relative pointer containing `..` must be lexically expanded against the catalog parent without a containment rejection or asset lookup.
- A failed overwrite of an existing remote catalog must not delete that pre-existing object.
- An unsupported remote suffix must fail before any destination object is created.
- Explicit storage options must reach fsspec without appearing on `GeoVector` or in any persisted property.

---

## File Map

- Create `src/geosave_engine/geodata/utils/io/storage.py`: provider-neutral filesystem resolution and local/remote asset-reference arithmetic.
- Create `tests/geodata/utils/io/test_storage.py`: focused storage-boundary tests with local and in-memory filesystems.
- Modify `src/geosave_engine/geodata/core/vector.py`: remove persistence source state and the row-level resolver; expose remote return typing and examples.
- Modify `src/geosave_engine/geodata/utils/io/__init__.py`: dispatch vector reads and materialize GeoParquet asset pointers at the read boundary.
- Modify `src/geosave_engine/geodata/utils/io/geoparquet.py`: normalize pointer values, pass native fsspec filesystems to GeoPandas, and retain distinct local/remote write guarantees.
- Modify `tests/geodata/test_vector.py`: replace source/resolver expectations with materialized pointer behavior.
- Modify `tests/geodata/test_raster_io_smoke.py`: remove source-state assertions while retaining local filtering and atomic-staging coverage.
- Create `tests/geodata/test_remote_geoparquet.py`: in-memory remote round trips, filtering, overwrite/failure semantics, and the marked Hugging Face bucket check.
- Modify `pyproject.toml`: declare fsspec directly.
- Modify `uv.lock`: record the direct dependency without changing the resolved fsspec version unnecessarily.

### Task 1: Add the Filesystem and Location Boundary

**Files:**
- Create: `src/geosave_engine/geodata/utils/io/storage.py`
- Create: `tests/geodata/utils/io/test_storage.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: fsspec `url_to_fs`, `AbstractFileSystem`, `LocalFileSystem`, and `split_protocol`.
- Produces: `StorageOptions`, `filesystem_path(location, storage_options) -> tuple[AbstractFileSystem, str]`, `is_local_filesystem(filesystem) -> TypeGuard[LocalFileSystem]`, `resolve_asset_path(reference, *, filesystem, catalog_path) -> Path | str`, and `stored_asset_path(reference, *, filesystem, catalog_path) -> str`.

- [ ] **Step 1: Write failing filesystem and path-arithmetic tests**

Create `tests/geodata/utils/io/test_storage.py` with these cases:

```python
from pathlib import Path

import fsspec
from fsspec.spec import AbstractFileSystem

from geosave_engine.geodata.utils.io.storage import (
    filesystem_path,
    resolve_asset_path,
    stored_asset_path,
)


def test_filesystem_path_forwards_storage_options(monkeypatch) -> None:
    memory = fsspec.filesystem("memory")
    received: dict[str, object] = {}

    def resolve(
        location: str, **options: object
    ) -> tuple[AbstractFileSystem, str]:
        received["location"] = location
        received["options"] = options
        return memory, "/catalogs/catalog.parquet"

    monkeypatch.setattr(fsspec.core, "url_to_fs", resolve)

    result = filesystem_path(
        "hf://buckets/fatmur/test/catalog.parquet",
        {"token": "test-token"},
    )

    assert result == (memory, "/catalogs/catalog.parquet")
    assert received == {
        "location": "hf://buckets/fatmur/test/catalog.parquet",
        "options": {"token": "test-token"},
    }


def test_remote_asset_paths_round_trip_without_contacting_the_asset() -> None:
    filesystem, catalog_path = filesystem_path(
        "memory://dataset/catalog.parquet"
    )

    assert stored_asset_path(
        "memory://dataset/rasters/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "rasters/prediction.zarr"
    assert resolve_asset_path(
        "rasters/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "memory:///dataset/rasters/prediction.zarr"
    assert stored_asset_path(
        "s3://other-bucket/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "s3://other-bucket/prediction.zarr"


def test_parent_relative_asset_is_allowed_without_asset_lookup() -> None:
    filesystem, catalog_path = filesystem_path(
        "memory://dataset/catalogs/catalog.parquet"
    )

    assert resolve_asset_path(
        "../shared/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "memory:///dataset/shared/prediction.zarr"


def test_local_asset_paths_are_lexical_and_portable(tmp_path: Path) -> None:
    catalog = tmp_path / "dataset" / "catalog.parquet"
    inside = tmp_path / "dataset" / "rasters" / "prediction.zarr"
    outside = tmp_path / "shared" / "prediction.zarr"
    filesystem, catalog_path = filesystem_path(catalog)

    assert stored_asset_path(
        inside, filesystem=filesystem, catalog_path=catalog_path
    ) == "rasters/prediction.zarr"
    assert stored_asset_path(
        outside, filesystem=filesystem, catalog_path=catalog_path
    ) == str(outside)
    assert resolve_asset_path(
        "../shared/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == outside
```

- [ ] **Step 2: Run the focused test and confirm the missing module is the failure**

Run: `uv run pytest tests/geodata/utils/io/test_storage.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'geosave_engine.geodata.utils.io.storage'`.

- [ ] **Step 3: Declare fsspec as an owned runtime dependency**

Add the existing resolved version as a direct dependency in `pyproject.toml`:

```toml
dependencies = [
    "fsspec>=2026.3.0",
]
```

Keep the rest of the dependency list intact, then run `uv lock`. Confirm the lock diff changes fsspec's dependency ownership only and does not unexpectedly upgrade unrelated packages.

- [ ] **Step 4: Implement the storage functions without provider branches**

Create `src/geosave_engine/geodata/utils/io/storage.py`:

```python
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


def filesystem_path(
    location: str | PathLike[str],
    storage_options: StorageOptions | None = None,
) -> tuple[AbstractFileSystem, str]:
    """Resolve a location to its native filesystem and protocol-free path."""
    return fsspec.core.url_to_fs(
        str(location), **dict(storage_options or {})
    )


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
        parent = PurePosixPath(
            posixpath.dirname(catalog_path).lstrip("/")
        )
        candidate = PurePosixPath(
            posixpath.normpath(protocol_path).lstrip("/")
        )
        try:
            return candidate.relative_to(parent).as_posix()
        except ValueError:
            return spelled

    candidate = Path(spelled)
    if not candidate.is_absolute():
        return candidate.as_posix()
    if not is_local_filesystem(filesystem):
        return spelled

    parent = Path(catalog_path).parent
    normalized = Path(os.path.abspath(candidate))
    try:
        return normalized.relative_to(parent).as_posix()
    except ValueError:
        return spelled
```

Keep the functions lexical: do not call `exists`, `resolve`, `open`, `ls`, or any provider API while manipulating an asset reference.

- [ ] **Step 5: Run storage tests and static checks**

Run:

```bash
uv run pytest tests/geodata/utils/io/test_storage.py -v
uv run ruff check src/geosave_engine/geodata/utils/io/storage.py tests/geodata/utils/io/test_storage.py
uv run basedpyright src/geosave_engine/geodata/utils/io/storage.py tests/geodata/utils/io/test_storage.py
```

Expected: all tests and checks pass. If the monkeypatched resolver needs an explicit return annotation for BasedPyright, annotate it as `tuple[AbstractFileSystem, str]` and import `AbstractFileSystem` in the test.

- [ ] **Step 6: Commit only the isolated storage boundary when safe**

Inspect `git diff -- pyproject.toml uv.lock` first. If those files contain unrelated pre-existing edits, leave this task uncommitted. Otherwise run:

```bash
git add src/geosave_engine/geodata/utils/io/storage.py tests/geodata/utils/io/test_storage.py pyproject.toml uv.lock
git commit --only src/geosave_engine/geodata/utils/io/storage.py tests/geodata/utils/io/test_storage.py pyproject.toml uv.lock -m "feat: add geodata storage boundary"
```

### Task 2: Make GeoVector Stateless and Materialize Catalog Pointers

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `src/geosave_engine/geodata/utils/io/__init__.py`
- Modify: `tests/geodata/test_vector.py`
- Modify: `tests/geodata/test_raster_io_smoke.py`

**Interfaces:**
- Consumes: Task 1 `StorageOptions`, `filesystem_path`, and `resolve_asset_path`.
- Produces: `GeoVector(gdf)` with no `source` field or `resolve_path`; `read_vector` returns GeoParquet rows whose non-null `path` values are directly usable `Path | str` pointers.

- [ ] **Step 1: Replace source-state tests with read-boundary pointer tests**

In `tests/geodata/test_vector.py`, replace the movable-catalog, containment, and resolver tests with:

```python
def test_asset_path_round_trips_relative_to_movable_catalog(tmp_path: Path) -> None:
    original = tmp_path / "original"
    (original / "rasters" / "prediction.zarr").mkdir(parents=True)
    catalog = GeoVector.from_geometry(
        Point(0, 0), path=original / "rasters" / "prediction.zarr"
    )

    catalog.to_geoparquet(original / "catalog.parquet")
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    restored = read_vector(moved / "catalog.parquet")

    assert restored.gdf.iloc[0].path == moved / "rasters" / "prediction.zarr"
    assert not hasattr(restored, "source")


def test_absolute_asset_outside_catalog_remains_absolute(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "shared" / "prediction.zarr"
    catalog = GeoVector.from_geometry(Point(0, 0), path=outside)

    written = catalog.to_geoparquet(root / "catalog.parquet")
    restored = read_vector(written)

    assert restored.gdf.iloc[0].path == outside


def test_relative_parent_asset_expands_without_dereferencing(tmp_path: Path) -> None:
    root = tmp_path / "dataset" / "catalogs"
    root.mkdir(parents=True)
    catalog = GeoVector.from_geometry(
        Point(0, 0), path="../shared/missing.zarr"
    )

    written = catalog.to_geoparquet(root / "catalog.parquet")
    restored = read_vector(written)

    assert restored.gdf.iloc[0].path == tmp_path / "dataset" / "shared" / "missing.zarr"


def test_unsaved_vector_keeps_the_caller_pointer() -> None:
    vector = GeoVector.from_geometry(Point(0, 0), path="assets/a.zarr")

    assert vector.gdf.iloc[0].path == "assets/a.zarr"
    assert not hasattr(vector, "resolve_path")
```

Update `test_catalog_register_query_upsert_and_relocate` so its final assertion is:

```python
assert asset_rows.iloc[0].path == asset
```

In `tests/geodata/test_raster_io_smoke.py`, delete `assert selected.source == path` and retain the bbox/column assertions.

- [ ] **Step 2: Run the pointer tests and confirm they fail on source-era behavior**

Run:

```bash
uv run pytest tests/geodata/test_vector.py -k "asset or catalog_register" -v
uv run pytest tests/geodata/test_raster_io_smoke.py -k geoparquet -v
```

Expected: the movable and external pointer assertions fail because `read_vector` still returns stored relative values, and the API-removal assertions fail because `source` and `resolve_path` still exist.

- [ ] **Step 3: Remove persistence state from GeoVector**

In `src/geosave_engine/geodata/core/vector.py`:

- remove `Mapping`, `field`, and `isnan` imports that existed only for `resolve_path`;
- remove `source` from the dataclass;
- delete `resolve_path` entirely;
- construct plain `GeoVector` values in `to_crs`, `upsert`, and `query`.

The affected returns become:

```python
return type(self)(self.gdf.to_crs(crs))
```

```python
retained = type(self)(
    self.gdf.loc[~self.gdf[on].isin(incoming)].copy()
)
combined = type(self).concat([retained, records])
return type(self)(combined.gdf)
```

```python
if self.gdf.empty:
    return type(self)(self.gdf.copy())

return type(self)(candidates.loc[exact].copy())
```

Change the upsert return text to `Updated collection.`; no in-memory operation owns persistence context.

- [ ] **Step 4: Expand GeoParquet pointers in read_vector**

In `src/geosave_engine/geodata/utils/io/__init__.py`, import `PathLike` at runtime plus the Task 1 helpers. Replace the source-bearing dispatch with:

```python
if suffix in (".geojson", ".json"):
    return GeoVector(geojson.read(source, **options))

if suffix == ".gpkg":
    return GeoVector(geopackage.read(source, **options))

if suffix in (".parquet", ".geoparquet"):
    frame = geoparquet.read(source, **options)
    if "path" in frame:
        storage_options = cast(
            "StorageOptions | None", options.get("storage_options")
        )
        filesystem, catalog_path = filesystem_path(
            source, storage_options=storage_options
        )
        values = frame["path"].dropna()
        invalid = [
            value
            for value in values
            if not isinstance(value, (str, PathLike))
        ]
        if invalid:
            value = invalid[0]
            raise TypeError(
                "asset path must be string or path-like, got "
                f"{type(value).__name__}"
            )
        frame.loc[values.index, "path"] = [
            resolve_asset_path(
                value,
                filesystem=filesystem,
                catalog_path=catalog_path,
            )
            for value in values
        ]
    return GeoVector(frame)
```

Import `cast` from `typing`, and import `StorageOptions`, `filesystem_path`, and `resolve_asset_path` from `.storage`. Use the exact narrow cast below rather than broadening any storage function to `Any`:

```python
storage_options = cast(
    "StorageOptions | None", options.get("storage_options")
)
```

- [ ] **Step 5: Make local pointer storage preserve external references**

Until Task 3 moves normalization into the complete local/remote writer, update `portable_paths` in `geoparquet.py` to use `filesystem_path` and `stored_asset_path` instead of `Path.resolve` plus containment rejection. Its public behavior must be:

```python
filesystem, catalog_path = filesystem_path(destination)
frame.loc[values.index, "path"] = [
    stored_asset_path(
        value,
        filesystem=filesystem,
        catalog_path=catalog_path,
    )
    for value in values
]
```

Validate all non-null values as `str | PathLike[str]` before mapping. Delete the `ValueError` containment branch; outside references are valid pointers.

- [ ] **Step 6: Run the complete GeoVector and local persistence tests**

Run:

```bash
uv run pytest tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py -k "vector or geoparquet or catalog or asset" -v
uv run ruff check src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/__init__.py src/geosave_engine/geodata/utils/io/geoparquet.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py
```

Expected: the selected tests pass, local paths inside the catalog become relative on disk and absolute when read, and paths outside the catalog remain absolute.

- [ ] **Step 7: Commit the stateless vector boundary when safe**

Inspect the four modified files for pre-existing work first. If every hunk belongs to this feature, run:

```bash
git add src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/__init__.py src/geosave_engine/geodata/utils/io/geoparquet.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py
git commit --only src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/__init__.py src/geosave_engine/geodata/utils/io/geoparquet.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py -m "refactor: resolve vector assets at persistence boundary"
```

Otherwise leave the task changes uncommitted and record which paths prevented an isolated commit.

### Task 3: Read and Write GeoParquet Through Native Filesystems

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `src/geosave_engine/geodata/utils/io/geoparquet.py`
- Create: `tests/geodata/test_remote_geoparquet.py`

**Interfaces:**
- Consumes: Task 1 filesystem helpers and Task 2 stateless/materialized pointer contract.
- Produces: `GeoParquetOpenOptions.storage_options`, `GeoParquetWriteOptions.storage_options`, `geoparquet.write(...) -> Path | str`, and `GeoVector.to_geoparquet(...) -> Path | str`.

- [ ] **Step 1: Write failing in-memory GeoParquet tests**

Create `tests/geodata/test_remote_geoparquet.py` with these unit cases:

```python
from uuid import uuid4

import fsspec
import geopandas as gpd
import pytest
from fsspec.spec import AbstractFileSystem
from shapely.geometry import Point, box

from geosave_engine.geodata import GeoVector, read_vector


def memory_catalog() -> str:
    return f"memory://geosave-tests/{uuid4()}/catalog.parquet"


def test_remote_geoparquet_round_trip_filters_and_materializes_path() -> None:
    destination = memory_catalog()
    parent = destination.rsplit("/", 1)[0]
    vector = GeoVector(
        gpd.GeoDataFrame(
            {
                "name": ["near", "far"],
                "path": [f"{parent}/rasters/near.zarr", None],
            },
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs="EPSG:4326",
        )
    )

    written = vector.to_geoparquet(
        destination, write_covering_bbox=True
    )
    selected = read_vector(
        destination,
        bbox=(-1, -1, 2, 2),
        columns=["name", "path", "geometry"],
    )

    filesystem, target = fsspec.core.url_to_fs(destination)
    expected_asset = filesystem.unstrip_protocol(
        f"{target.rsplit('/', 1)[0]}/rasters/near.zarr"
    )
    assert written == destination
    assert list(selected.gdf.name) == ["near"]
    assert selected.gdf.iloc[0].path == expected_asset
    assert "token" not in selected.gdf
    assert not hasattr(selected, "source")


def test_external_remote_asset_does_not_require_its_driver() -> None:
    destination = memory_catalog()
    external = "s3://other-bucket/prediction.zarr"

    GeoVector.from_geometry(Point(0, 0), path=external).to_geoparquet(
        destination
    )

    assert read_vector(destination).gdf.iloc[0].path == external


def test_null_asset_pointer_remains_null() -> None:
    destination = memory_catalog()
    vector = GeoVector(
        gpd.GeoDataFrame(
            {"path": [None]}, geometry=[Point(0, 0)], crs="EPSG:4326"
        )
    )

    vector.to_geoparquet(destination)

    assert read_vector(destination).gdf.path.isna().all()


def test_opaque_asset_pointer_fails_before_writing() -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)
    vector = GeoVector.from_geometry(Point(0, 0), path=object())

    with pytest.raises(TypeError, match="asset path must be string or path-like"):
        vector.to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_remote_write_refuses_existing_catalog() -> None:
    destination = memory_catalog()
    vector = GeoVector.from_geometry(Point(0, 0))
    vector.to_geoparquet(destination)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        vector.to_geoparquet(destination)


def test_invalid_remote_suffix_creates_nothing() -> None:
    destination = memory_catalog().replace(".parquet", ".json")
    filesystem, target = fsspec.core.url_to_fs(destination)

    with pytest.raises(ValueError, match="must end"):
        GeoVector.from_geometry(Point(0, 0)).to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_failed_new_remote_write_removes_its_partial_object(monkeypatch) -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)

    def fail(_self, path: str, **options: object) -> None:
        remote = options["filesystem"]
        assert isinstance(remote, AbstractFileSystem)
        remote.pipe(path, b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)

    with pytest.raises(RuntimeError, match="encode failed"):
        GeoVector.from_geometry(Point(0, 0)).to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_failed_remote_overwrite_does_not_delete_existing_object(
    monkeypatch,
) -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)
    filesystem.pipe(target, b"existing")

    def fail(_self, path: str, **options: object) -> None:
        raise RuntimeError("encode failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)

    with pytest.raises(RuntimeError, match="encode failed"):
        GeoVector.from_geometry(Point(0, 0)).to_geoparquet(
            destination, overwrite=True
        )

    assert filesystem.cat(target) == b"existing"
```

- [ ] **Step 2: Run the remote tests and confirm local-only writer failures**

Run: `uv run pytest tests/geodata/test_remote_geoparquet.py -v`

Expected: tests fail because the writer treats the URL as a local `Path`, `GeoParquetWriteOptions` does not accept `storage_options`, and the reader does not pass an fsspec filesystem explicitly.

- [ ] **Step 3: Resolve reads once and pass the native filesystem to GeoPandas**

In `geoparquet.py`, narrow storage typing and change `read` to:

```python
class GeoParquetOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when opening GeoParquet."""

    columns: Sequence[str] | None
    bbox: tuple[float, float, float, float] | None
    storage_options: StorageOptions | None
    to_pandas_kwargs: Mapping[str, Any] | None
    parquet_options: dict[str, Any]


def read(
    source: str | PathLike[str],
    **open_options: Unpack[GeoParquetOpenOptions],
) -> gpd.GeoDataFrame:
    """Open one local or fsspec-backed GeoParquet file."""
    storage_options = open_options.pop("storage_options", None)
    parquet_options = dict(open_options.pop("parquet_options", {}))
    filesystem, source_path = filesystem_path(source, storage_options)
    return gpd.read_parquet(
        source_path,
        filesystem=filesystem,
        **open_options,
        **parquet_options,
    )
```

Keep `columns`, `bbox`, and `to_pandas_kwargs` flowing to GeoPandas so PyArrow can retain projection and covering-bbox filtering.

- [ ] **Step 4: Implement separate local and remote write guarantees**

Add `storage_options: StorageOptions | None` to `GeoParquetWriteOptions`. Replace `write` with the following control flow:

```python
def write(
    gdf: gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    **write_options: Unpack[GeoParquetWriteOptions],
) -> Path | str:
    """Write one GeoDataFrame to local or fsspec-backed GeoParquet."""
    storage_options = write_options.pop("storage_options", None)
    filesystem, target = filesystem_path(path, storage_options)
    suffix = PurePosixPath(target).suffix.lower()
    if suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {PurePosixPath(target).name!r} must end in one of "
            f"{list(_FILE_SUFFIXES)}"
        )

    frame = gdf
    if "path" in gdf:
        frame = gdf.copy()
        values = frame["path"].dropna()
        invalid = [
            value
            for value in values
            if not isinstance(value, (str, PathLike))
        ]
        if invalid:
            value = invalid[0]
            raise TypeError(
                "asset path must be string or path-like, got "
                f"{type(value).__name__}"
            )
        frame.loc[values.index, "path"] = [
            stored_asset_path(
                value,
                filesystem=filesystem,
                catalog_path=target,
            )
            for value in values
        ]

    parquet_options = dict(write_options.pop("parquet_options", {}))
    if is_local_filesystem(filesystem):
        local_target = Path(target)
        if local_target.exists() and not overwrite:
            raise FileExistsError(
                f"{local_target} exists; pass overwrite=True to replace it"
            )
        staged = local_target.with_name(
            f".{local_target.stem}.staging{local_target.suffix}"
        )
        try:
            frame.to_parquet(
                staged, **write_options, **parquet_options
            )
            os.replace(staged, local_target)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        return Path(path)

    existed = filesystem.exists(target)
    if existed and not overwrite:
        raise FileExistsError(
            f"{path} exists; pass overwrite=True to replace it"
        )
    try:
        frame.to_parquet(
            target,
            filesystem=filesystem,
            **write_options,
            **parquet_options,
        )
    except BaseException:
        if not existed:
            with suppress(Exception):
                filesystem.rm(target)
        raise
    return str(path)
```

Import `suppress`, `PurePosixPath`, and the Task 1 functions/types. Delete `portable_paths`: normalization now occurs in `write` after the destination filesystem has been resolved, so `GeoVector.to_geoparquet` must pass `self.gdf` directly. Retain the existing local stale-staging test.

- [ ] **Step 5: Update the public return type and remote example**

In `GeoVector.to_geoparquet`, change the return annotation to `Path | str`, call `geoparquet.write(self.gdf, ...)`, and add this example without logging a token:

```python
>>> catalog.to_geoparquet(
...     "hf://buckets/fatmur/test/catalog.parquet",
...     storage_options={"token": token},
... )
'hf://buckets/fatmur/test/catalog.parquet'
```

Document that local writes return `Path` and URL writes return `str`.

- [ ] **Step 6: Run remote, local, filtering, and type checks**

Run:

```bash
uv run pytest tests/geodata/test_remote_geoparquet.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py -k "geoparquet or asset or catalog" -v
uv run ruff check src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/storage.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py tests/geodata/utils/io/test_storage.py tests/geodata/test_remote_geoparquet.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py
uv run basedpyright src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/storage.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py
uv run mypy src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/storage.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py
```

Expected: unit tests pass, local output remains a `Path`, remote output preserves the supplied URL string, and static checks introduce no new errors.

- [ ] **Step 7: Commit remote GeoParquet support when safe**

Inspect all listed paths. If their full diffs belong to this plan, run:

```bash
git add src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py tests/geodata/test_remote_geoparquet.py
git commit --only src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py tests/geodata/test_remote_geoparquet.py -m "feat: persist geoparquet through fsspec"
```

Otherwise preserve the changes uncommitted and report the overlapping paths.

### Task 4: Verify the Hugging Face Bucket and the Geodata Regression Surface

**Files:**
- Modify: `tests/geodata/test_remote_geoparquet.py`
- Modify: `src/geosave_engine/geodata/utils/io/__init__.py`

**Interfaces:**
- Consumes: public `GeoVector.to_geoparquet(..., storage_options=...) -> Path | str`, `read_vector(..., storage_options=...) -> GeoVector`, and Hugging Face `get_token`/`HfFileSystem` authentication defaults.
- Produces: one opt-in integration proof against `hf://buckets/fatmur/test` and finalized public read documentation.

- [ ] **Step 1: Add the authenticated bucket round-trip test**

Append to `tests/geodata/test_remote_geoparquet.py`:

```python
from huggingface_hub import get_token


@pytest.mark.integration
def test_huggingface_bucket_geoparquet_round_trip() -> None:
    token = get_token()
    if token is None:
        pytest.skip("HF_TOKEN or `hf auth login` is required")

    prefix = (
        "hf://buckets/fatmur/test/geosave-tests/"
        f"{uuid4()}"
    )
    destination = f"{prefix}/catalog.parquet"
    filesystem, prefix_path = fsspec.core.url_to_fs(prefix)
    vector = GeoVector(
        gpd.GeoDataFrame(
            {"name": ["near", "far"]},
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs="EPSG:4326",
        )
    )

    try:
        vector.to_geoparquet(
            destination,
            write_covering_bbox=True,
        )
        selected = read_vector(
            destination,
            bbox=(-1, -1, 2, 2),
            columns=["name", "geometry"],
        )

        assert list(selected.gdf.name) == ["near"]
    finally:
        if filesystem.exists(prefix_path):
            filesystem.rm(prefix_path, recursive=True)
```

This live check intentionally omits `storage_options`, proving that `HF_TOKEN` and `hf auth login` remain owned by `HfFileSystem`; Task 1 separately proves explicit options are forwarded. The cleanup target is the UUID-specific prefix, never `fatmur/test` or the shared `geosave-tests` parent. Do not print the filesystem repr or token.

- [ ] **Step 2: Document URL reads at the public dispatcher**

Update the `read_vector` docstring to say that GeoParquet accepts local paths or fsspec URLs and that relative `path` properties become directly usable pointers. Add:

```python
>>> catalog = read_vector(
...     "hf://buckets/fatmur/test/catalog.parquet",
...     storage_options={"token": token},
... )
>>> catalog.query(prediction).gdf.iloc[0]["path"]
'hf://buckets/fatmur/test/rasters/prediction.zarr'
```

Do not imply remote support for GeoJSON or GeoPackage.

- [ ] **Step 3: Run the live integration test when authentication is available**

Run:

```bash
uv run pytest -m integration tests/geodata/test_remote_geoparquet.py::test_huggingface_bucket_geoparquet_round_trip -v
```

Expected with `HF_TOKEN` or `hf auth login` access to `fatmur/test`: PASS and the UUID prefix is absent afterward. Expected without authentication: SKIP with the explicit authentication message. A skip verifies unit behavior only; report live bucket verification as outstanding rather than claiming it passed.

- [ ] **Step 4: Run the full relevant verification matrix**

Run:

```bash
uv run pytest tests/geodata/utils/io/test_storage.py tests/geodata/test_remote_geoparquet.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py -v
uv run pytest tests/geodata -v
uv run ruff check src/geosave_engine/geodata tests/geodata
uv run basedpyright src/geosave_engine/geodata
uv run mypy src/geosave_engine/geodata
```

Expected: focused and geodata tests pass; lint and type checks pass. If the known unrelated missing `geosave_engine.workflow.examples.reflectance` collection error is still present outside `tests/geodata`, record it separately and do not attribute it to this feature.

- [ ] **Step 5: Inspect the final diff for architecture regressions**

Run:

```bash
git diff --check
git diff -- src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/storage.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py tests/geodata/utils/io/test_storage.py tests/geodata/test_remote_geoparquet.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py pyproject.toml uv.lock
```

Confirm the diff contains no `GeoVector.source`, `resolve_path`, provider switch, stored credential, asset upload, `s3fs` dependency, or remote raster-format implementation.

- [ ] **Step 6: Commit the integration proof when safe**

If the two files contain only this plan's changes, run:

```bash
git add tests/geodata/test_remote_geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py
git commit --only tests/geodata/test_remote_geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py -m "test: verify hugging face geoparquet persistence"
```

If either file overlaps pre-existing uncommitted work, leave the final changes uncommitted and include that fact in the handoff.
