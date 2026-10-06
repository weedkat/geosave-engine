# Remote Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let raster writers, readers and catalog registration take an fsspec
URL and `storage_options`, the way GeoParquet tables already do.

**Architecture:** A location is a local path or an fsspec URL, resolved by
`storage.filesystem_path`. Zarr goes through fsspec directly. COG and NetCDF
write a local file that `storage.local_target` uploads. TIFF reads hand the
resolved filesystem to rasterio as its `opener` when `storage_options` is given.

**Tech Stack:** fsspec, xarray, rasterio 1.5, zarr 3, pytest with fsspec's
`memory://` filesystem.

**Spec:** `docs/superpowers/specs/2026-10-06-remote-storage-design.md`.

## Global Constraints

- The checkout is dirty on `main` and shared. Do not commit, stash, checkout or restore.
- No new dependencies and no path wrapper class. `storage_options` is the only
  credentials argument.
- A URL a format cannot handle raises; it never becomes a local folder.
- A local write returns `Path`; a remote write returns the URL `str` it was given.
- No private helper beyond what this plan shows. If a step needs more code or
  branches than shown, or a second path for one job, stop and report.
- Readers must not compute pixels while opening.
- Google-style docstrings checked with `python scripts/check_docstrings.py`.
- Remote tests use `memory://` under a unique prefix and remove it afterwards.

## Review Focus

1. A URL with a trailing slash, and a name holding a dot (`s3://b/scene.v2`):
   joined names keep `scheme://` and the dot (Task 3).
2. A failed remote write uploads nothing and leaves an existing object alone (Task 1).
3. `overwrite=False` on an existing remote object raises before any local work (Task 1).
4. A Zarr URL passed to a writer never creates a local folder named after the
   scheme (Task 2).
5. Catalog hrefs for remote assets stay absolute URLs when the table lives on
   another filesystem, and become relative when it shares one (Task 5).

## File Structure

| File | Change |
| --- | --- |
| `geodata/io/storage.py` | `local_target` |
| `geodata/io/zarr.py` | URL destinations, `storage_options` on write |
| `geodata/io/geotiff.py`, `geodata/io/cogs.py` | upload through `local_target`; URL-safe names |
| `geodata/io/gdal.py` | `storage_options` and rasterio `opener` |
| `geodata/io/netcdf.py` | upload on write; URL read refused |
| `geodata/stac/asset.py`, `stac/item.py`, `core/vector.py` | forward `storage_options` |
| `geodata/core/{raster,stack,array}.py` | `storage_options` on the `gs` writers |

---

### Task 1: `storage.local_target`

**Files:**
- Modify: `src/geosave_engine/geodata/io/storage.py`
- Test: `tests/geodata/io/test_storage.py`

**Interfaces:**
- Produces:
  `local_target(location, *, overwrite=False, storage_options=None) -> ContextManager[Path]`.

- [x] **Step 1: Write the failing tests**

```python
import uuid
from pathlib import Path

import pytest

from geosave_engine.geodata.io.storage import local_target


@pytest.fixture
def bucket():
    """Yield a unique `memory://` prefix and empty it afterwards."""
    filesystem = fsspec.filesystem("memory")
    prefix = f"memory://geosave-tests/{uuid.uuid4()}"
    yield prefix
    filesystem.rm(prefix.removeprefix("memory:/"), recursive=True)


def test_a_local_location_is_its_own_target(tmp_path: Path) -> None:
    with local_target(tmp_path / "scene.tif") as target:
        assert target == tmp_path / "scene.tif"


def test_a_remote_location_uploads_what_was_written(bucket: str) -> None:
    with local_target(f"{bucket}/scene.tif") as target:
        assert target.name == "scene.tif"
        target.write_bytes(b"pixels")

    assert fsspec.open(f"{bucket}/scene.tif").open().read() == b"pixels"
    assert not target.exists()


def test_a_failed_write_uploads_nothing(bucket: str) -> None:
    with pytest.raises(RuntimeError), local_target(f"{bucket}/scene.tif") as target:
        target.write_bytes(b"partial")
        raise RuntimeError("writer failed")

    assert not fsspec.filesystem("memory").exists(f"{bucket}/scene.tif")


def test_an_existing_remote_object_refuses_without_overwrite(bucket: str) -> None:
    with local_target(f"{bucket}/scene.tif") as target:
        target.write_bytes(b"first")

    with pytest.raises(FileExistsError), local_target(f"{bucket}/scene.tif"):
        pytest.fail("the body must not run")
    with local_target(f"{bucket}/scene.tif", overwrite=True) as target:
        target.write_bytes(b"second")
    assert fsspec.open(f"{bucket}/scene.tif").open().read() == b"second"
```

Move the `bucket` fixture to `tests/geodata/io/conftest.py` (create it if
absent) so later tasks share it.

- [x] **Step 2: Run** `uv run pytest tests/geodata/io/test_storage.py -q`
  Expected: FAIL with `ImportError: cannot import name 'local_target'`.

- [x] **Step 3: Implement** in `io/storage.py`:

```python
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from tempfile import TemporaryDirectory


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
        overwrite: Replace an existing remote object.
        storage_options: Options for the location's filesystem.

    Yields:
        `location` itself when it is local, else a path of the same name in a
        temporary folder, uploaded once the block ends without an error.

    Raises:
        FileExistsError: The remote object exists and `overwrite` is false.

    Examples:
        >>> with local_target("s3://bucket/scene.tif") as target:
        ...     write(target)
    """
    filesystem, path = filesystem_path(location, storage_options)
    if is_local_filesystem(filesystem):
        yield Path(path)
        return
    if not overwrite and filesystem.exists(path):
        raise FileExistsError(f"{location} exists; pass overwrite=True to replace it")
    with TemporaryDirectory() as folder:
        target = Path(folder) / PurePosixPath(path).name
        yield target
        filesystem.put(str(target), path)
```

Add `from pathlib import Path, PurePosixPath`.

- [x] **Step 4: Run** `uv run pytest tests/geodata/io/test_storage.py -q`
  Expected: PASS.

---

### Task 2: Zarr writes to a URL

**Files:**
- Modify: `src/geosave_engine/geodata/io/zarr.py` (`write`), `core/raster.py` and `core/stack.py` (`to_zarr`)
- Test: `tests/geodata/io/test_zarr.py`

**Interfaces:**
- Produces: `zarr.write(raster_or_stack, destination, *, compute=True, overwrite=False, storage_options=None, **write_options) -> Path | str | Delayed`;
  `to_zarr(destination, *, compute=True, overwrite=False, storage_options=None, **write_options)` on `GeoRaster` and `GeoStack`.

- [x] **Step 1: Write the failing tests**

```python
def test_a_raster_round_trips_through_a_remote_store(bucket, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    written = build_raster(times=2)
    destination = f"{bucket}/scene.zarr"

    saved = written.gs.to_zarr(destination, storage_options={})

    assert saved == destination
    assert list(tmp_path.iterdir()) == []
    with read_raster(destination, storage_options={}) as restored:
        xr.testing.assert_identical(restored, read_raster(written.gs.to_zarr(tmp_path / "local.zarr")))


def test_a_remote_store_refuses_to_be_replaced_silently(bucket) -> None:
    written = build_raster()
    written.gs.to_zarr(f"{bucket}/scene.zarr")

    with pytest.raises(FileExistsError):
        written.gs.to_zarr(f"{bucket}/scene.zarr")
    assert written.gs.to_zarr(f"{bucket}/scene.zarr", overwrite=True) == f"{bucket}/scene.zarr"


def test_a_remote_store_needs_the_zarr_suffix(bucket) -> None:
    with pytest.raises(ValueError, match=".zarr"):
        build_raster().gs.to_zarr(f"{bucket}/scene.store")
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/io/test_zarr.py -k remote -q`
  Expected: FAIL (`saved` is `memory:/...` and a local folder appears).

- [x] **Step 3: Implement.** In `zarr.write` add the `storage_options`
  keyword, document it, and replace `path = Path(destination)` and the two
  returns of `path`:

```python
    protocol, _ = split_protocol(str(destination))
    local = protocol in (None, "file", "local")
    # A URL passes to xarray as written; `Path` would fold its `scheme://`.
    store = Path(destination) if local else str(destination)
    if PurePosixPath(str(destination)).suffix != _STORE_SUFFIX:
        raise ValueError(
            f"destination {PurePosixPath(str(destination)).name!r} must end in "
            f"{_STORE_SUFFIX!r}"
        )
```

Pass `store` and `storage_options=storage_options` to both `to_zarr` calls and
return `store`. xarray's `mode="w-"` already raises `FileExistsError` on a
remote store; if the remote test shows another error type, stop and report.
Add the `storage_options` keyword to `GeoRaster.to_zarr` and `GeoStack.to_zarr`
with its `Args:` line, and forward it.

- [x] **Step 4: Run** `uv run pytest tests/geodata/io/test_zarr.py tests/geodata/io/test_zarr_stack.py tests/geodata/core -q`
  Expected: PASS.

---

### Task 3: COGs upload, and scene names survive a URL

**Files:**
- Modify: `src/geosave_engine/geodata/io/geotiff.py` (`write_cog`, `write_gtiff`, `_write`), `io/cogs.py` (`write`), `core/raster.py`, `core/stack.py`, `core/array.py` (`to_cog`, `to_gtiff`)
- Test: `tests/geodata/io/test_geotiff.py`, `tests/geodata/io/test_cogs.py`

**Interfaces:**
- Consumes: `local_target` (Task 1).
- Produces: `write_cog(..., storage_options=None) -> Path | str`,
  `cogs.write(..., storage_options=None) -> tuple[Path | str, ...]`.

- [x] **Step 1: Write the failing tests**

```python
# tests/geodata/io/test_geotiff.py
def test_a_cog_uploads_to_a_remote_location(bucket) -> None:
    written = build_raster(times=1).isel(time=0)

    saved = geotiff.write_cog(written, f"{bucket}/scene.tif", storage_options={})

    assert saved == f"{bucket}/scene.tif"
    filesystem = fsspec.filesystem("memory")
    with rasterio.open(saved.removeprefix("memory:/"), opener=filesystem.open) as src:
        assert src.descriptions == ("red", "nir")


# tests/geodata/io/test_cogs.py
@pytest.mark.parametrize("root", ["scene.v2", "scene.v2/"])
def test_remote_names_keep_their_scheme_and_dots(bucket, root) -> None:
    written = build_raster(times=2)

    paths = cogs.write(written, f"{bucket}/{root}", split_bands=True)

    assert paths == tuple(
        f"{bucket}/scene.v2/scene.v2_{day}/{band}.tif"
        for day in DAYS
        for band in ("red", "nir")
    )


def test_local_names_are_still_paths(tmp_path: Path) -> None:
    paths = cogs.write(build_raster(times=1), tmp_path / "scene")

    assert paths == (tmp_path / "scene" / "scene_20250601T000000.tif",)
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/io/test_geotiff.py tests/geodata/io/test_cogs.py -k "remote or still_paths" -q`
  Expected: the remote tests FAIL.

- [x] **Step 3: Implement.**

In `geotiff._write` add `storage_options`, and wrap the existing body from the
suffix check onward so it writes to the local target:

```python
    if PurePosixPath(str(path)).suffix.lower() not in _FILE_SUFFIXES:
        raise ValueError(...)  # existing message
    with local_target(path, overwrite=overwrite, storage_options=storage_options) as target:
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
        ...  # the existing body, unchanged, writing `target`
    local = split_protocol(str(path))[0] in (None, "file", "local")
    return Path(path) if local else str(path)
```

A local write returns its `Path` and a remote one the URL it was given, read
from the protocol as Task 2 does. `write_cog` and `write_gtiff` take and
forward `storage_options`.

In `cogs.write`, names are joined as text so a URL keeps its `scheme://`.
Replace the lines from `root = Path(path)` through the `cogs` list with:

```python
    root = str(path).rstrip("/")
    name = PurePosixPath(root).name
    if TIME_COORDINATE in ds.dims:
        ...  # existing validation of `times`
        scenes = [
            (f"{root}/{name}_{format_instant(instant)}", ds.isel({TIME_COORDINATE: index}))
            for index, instant in enumerate(times)
        ]
    else:
        scenes = [(root, ds)]

    split = split_bands and len(ds.data_vars) > 1
    cogs: list[tuple[str, xr.Dataset]] = []
    for scene_path, scene in scenes:
        if split:
            cogs.extend((f"{scene_path}/{band}.tif", scene[[band]]) for band in scene.data_vars)
        else:
            cogs.append((f"{scene_path}.tif", scene))
```

and pass `storage_options=storage_options` to each `write_cog`, which returns
`Path` or `str` per file. Add the keyword to the `gs` `to_cog` and `to_gtiff`
methods and forward it. `GeoStack.to_cog` joins the group name the same way:
`f"{str(destination).rstrip('/')}/{name}"`.

- [x] **Step 4: Run** `uv run pytest tests/geodata -q`
  Expected: PASS, including every existing local expectation in `test_cogs.py`.

---

### Task 4: Read a TIFF through `storage_options`; NetCDF upload

**Files:**
- Modify: `src/geosave_engine/geodata/io/gdal.py` (`read`), `io/netcdf.py` (`read`, `read_stack`, `write`), `core/raster.py`, `core/stack.py` (`to_netcdf`)
- Test: `tests/geodata/io/test_read_raster.py`, `tests/geodata/io/test_netcdf.py`

**Interfaces:**
- Consumes: `local_target`; `cogs.write` remote paths (Task 3).
- Produces: `gdal.read(source, *, chunks=None, mask_and_scale=False, storage_options=None, **open_options)`;
  `netcdf.write(..., storage_options=None) -> Path | str | Delayed`.

- [x] **Step 1: Write the failing tests**

```python
# tests/geodata/io/test_read_raster.py
@pytest.mark.parametrize("split_bands", [False, True])
def test_remote_cogs_read_back_lazily(bucket, tmp_path, split_bands) -> None:
    written = build_raster(times=2, packed=True)
    remote = cogs.write(written, f"{bucket}/scene", split_bands=split_bands)
    local = cogs.write(written, tmp_path / "scene", split_bands=split_bands)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        restored = read_raster(remote, chunks={}, storage_options={})

    assert started == []
    with read_raster(local, chunks={}) as expected:
        xr.testing.assert_identical(restored, expected)


# tests/geodata/io/test_netcdf.py
def test_netcdf_uploads_but_does_not_read_remotely(bucket, tmp_path) -> None:
    written = build_raster(times=2)

    saved = written.gs.to_netcdf(f"{bucket}/scene.nc", storage_options={})

    assert saved == f"{bucket}/scene.nc"
    fsspec.filesystem("memory").get(saved.removeprefix("memory:/"), str(tmp_path / "copy.nc"))
    with read_raster(tmp_path / "copy.nc") as restored:
        np.testing.assert_array_equal(restored.red, written.red)
    with pytest.raises(ValueError, match="local"):
        read_raster(saved, storage_options={})
    with pytest.raises(ValueError, match="compute"):
        written.gs.to_netcdf(f"{bucket}/later.nc", compute=False)
```

- [x] **Step 2: Run** those two tests. Expected: FAIL.

- [x] **Step 3: Implement.**

`gdal.read`: add `storage_options: StorageOptions | None = None` and replace
`with rasterio.open(source) as src:` with:

```python
    # GDAL opens local paths and its own remote schemes; a filesystem the
    # caller configures is handed to it as the opener instead.
    if storage_options is None:
        opened = rasterio.open(source)
    else:
        filesystem, path = filesystem_path(source, storage_options)
        opened = rasterio.open(path, opener=filesystem.open)
    with opened as src:
```

The rest of the body is unchanged; `cube.encoding["source"]` keeps
`absolute_location(source)`.

`netcdf.read` and `netcdf.read_stack`: accept `storage_options=None` and raise
when the source is a URL:

```python
    if split_protocol(str(source))[0] not in (None, "file", "local"):
        raise ValueError(
            f"{source} is remote, and NetCDF reads only local files; download "
            f"it first, or store the raster as Zarr"
        )
```

`netcdf.write`: add `storage_options`, check the suffix on
`PurePosixPath(str(destination))`, raise `ValueError` naming `compute` when
`compute=False` meets a URL, and run the existing eager write inside
`with local_target(destination, overwrite=overwrite, storage_options=storage_options) as path:`.
Return the `Path` for a local destination and `str(destination)` for a URL.
Add `storage_options` to `GeoRaster.to_netcdf` and `GeoStack.to_netcdf`.

- [x] **Step 4: Run** `uv run pytest tests/geodata -q`
  Expected: PASS.

---

### Task 5: Register and read a remote catalog

**Files:**
- Modify: `src/geosave_engine/geodata/stac/asset.py`, `stac/item.py`, `core/vector.py` (`from_rasters`)
- Test: `tests/geodata/stac/test_asset.py`, `tests/geodata/core/test_catalog.py`

**Interfaces:**
- Produces: `asset.from_path(path, *, storage_options=None)`,
  `item.from_paths(paths, *, storage_options=None)`,
  `GeoVector.from_rasters(paths, *, storage_options=None)`.

- [x] **Step 1: Write the failing tests**

```python
# tests/geodata/stac/test_asset.py
def test_a_remote_cog_is_described_from_its_header(bucket) -> None:
    saved = io.geotiff.write_cog(build_raster(times=1).isel(time=0), f"{bucket}/scene.tif")

    asset = from_path(saved, storage_options={})

    assert asset.href == saved
    assert asset.media_type == pystac.MediaType.COG
    assert [band["name"] for band in asset.to_dict()["eo:bands"]] == ["red", "nir"]


# tests/geodata/core/test_catalog.py
def test_the_loop_runs_on_a_bucket(bucket) -> None:
    source = build_raster(times=2, packed=True)
    paths = source.gs.to_cog(f"{bucket}/samples/forest", split_bands=True)

    catalog = GeoVector.from_rasters(paths, storage_options={})
    table = catalog.gs.to_geoparquet(f"{bucket}/samples/catalog.parquet", stac=True)
    restored = read_vector(table).gs.query(source).gs.to_raster(storage_options={})

    assert catalog["id"].tolist() == ["forest_20250601T000000", "forest_20250602T000000"]
    np.testing.assert_array_equal(restored.red, source.red)
    np.testing.assert_array_equal(restored.time, source.time)


def test_remote_assets_keep_urls_in_a_local_table(bucket, tmp_path) -> None:
    import pyarrow.parquet as pq

    paths = build_raster(times=1).gs.to_cog(f"{bucket}/forest")
    remote = GeoVector.from_rasters(paths, storage_options={})

    elsewhere = remote.gs.to_geoparquet(tmp_path / "catalog.parquet", stac=True)
    remote.gs.to_geoparquet(f"{bucket}/catalog.parquet", stac=True)
    beside = pq.read_table(
        f"{bucket}/catalog.parquet".removeprefix("memory:/"),
        filesystem=fsspec.filesystem("memory"),
    )

    # A table on another filesystem keeps the URL; one beside the assets shortens it.
    assert pq.read_table(elsewhere).column("assets")[0].as_py()["image"]["href"] == paths[0]
    assert beside.column("assets")[0].as_py()["image"]["href"] == (
        "./forest/forest_20250601T000000.tif"
    )
```

- [x] **Step 2: Run** those tests. Expected: FAIL (`storage_options` is not a parameter).

- [x] **Step 3: Implement.** `asset.from_path` takes `storage_options=None`,
  opens with `read_raster(path, storage_options=storage_options)` when it is
  given (plain `read_raster(path)` otherwise), and tells COG from GeoTIFF with
  the same open rule `gdal.read` uses. `item.from_paths` and
  `GeoVector.from_rasters` take and forward the keyword. Nothing else changes:
  hrefs are already kept as URLs and the id rule reads names from them.

  If `asset.from_path` ends up repeating `gdal.read`'s "GDAL or opener" branch,
  stop and report: that branch then belongs in one place both call.

- [x] **Step 4: Run** `uv run pytest tests/geodata tests/workflow tests/ml -q`
  Expected: PASS.

---

### Task 6: Docs and full verification

**Files:**
- Modify: `docs/guides/architecture.md`, `docs/guides/workflows.md` (where they describe storage)
- Test: one `@pytest.mark.integration` test in `tests/geodata/core/test_catalog.py` running `test_the_loop_runs_on_a_bucket`'s body on an `hf://` bucket, modelled on `test_huggingface_bucket_geoparquet_round_trip` in `tests/geodata/io/test_geoparquet.py`

- [x] **Step 1:** Add the integration test, skipped without a token the way the
  existing one is.
- [x] **Step 2:** Add the caller view from the spec to `docs/guides/architecture.md`,
  with the limits (NetCDF reads, folder scans, one `storage_options` per call).
- [x] **Step 3: Verify.**

```bash
uv run pytest
uv run ruff check src tests
python scripts/check_docstrings.py src/geosave_engine/geodata/io/storage.py \
  src/geosave_engine/geodata/io/zarr.py src/geosave_engine/geodata/io/geotiff.py \
  src/geosave_engine/geodata/io/cogs.py src/geosave_engine/geodata/io/gdal.py \
  src/geosave_engine/geodata/io/netcdf.py src/geosave_engine/geodata/stac/asset.py
ls -d memory:* 2>/dev/null   # must print nothing
```

Expected: tests and ruff pass, the docstring check is clean, and no folder
named after a URL scheme exists in the working tree. Report test counts and
every ruling.

---

## Execution notes, 2026-10-06

Executed inline in the shared dirty checkout; nothing was committed. Tasks 1 to
3 and the NetCDF half of Task 4 went as planned. The TIFF-read half of Task 4
and Task 5 changed; the spec's "Revision" section records why.

Rulings:

1. The `bucket` fixture lives in `tests/geodata/conftest.py`, not
   `tests/geodata/io/conftest.py`, because the catalog tests under `core` use it.
2. `storage_options` for Zarr went into `ZarrWriteOptions` and is forwarded to
   xarray as it is, so `zarr.write` and the `to_zarr` accessors gained no new
   keyword.
3. `local_target` owns the existence check for local and remote locations.
4. TIFF reads use GDAL-native access with `storage.gdal_path` mapping
   `hf://buckets/` to the S3 gateway; the rasterio `opener` route was removed.
5. Registration takes no `storage_options`. The planned `memory://` COG loop
   test became a Zarr loop on `memory://`, plus two `hf://` integration tests.
