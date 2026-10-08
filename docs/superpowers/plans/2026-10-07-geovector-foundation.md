# GeoVector Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GeoVector a plain spatial GeoDataFrame accessor, build STAC Items from an in-memory raster plus the paths its writer returned, and keep Items in a stac-geoparquet table owned by `geodata/stac`.

**Architecture:** New code is added first (`stac/asset.py`, `stac/item.py`, `stac/table.py`), then the accessors and callers switch to it, then the old path (`io/catalog.py`, the STAC half of GeoVector, STAC sniffing in vector I/O) is deleted. Each task leaves the suite green.

**Tech Stack:** xarray, GeoPandas, PySTAC, stac-geoparquet, odc-geo, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-07-geovector-foundation-design.md`

## Global Constraints

- Do not commit. The working tree holds unrelated uncommitted changes; the user commits. Every "Checkpoint" step runs tests only.
- Never `git checkout`, `git restore` or `git stash` a file. Edit in place.
- No compatibility aliases. A removed name is removed.
- No new dependency. `pyproject.toml` is not edited.
- `geodata` imports no torch. `geodata/io` imports nothing from `geodata/stac`.
- Public functions get concise Google-style docstrings. Comments say what a block does or name a domain constraint, never how or why it changed.
- Names stay short: `name`, `paths`, `hrefs`, `rows`, `items`.
- Stop and report to the user, instead of pushing on, when: a step needs code that is not in this plan, a function grows past about 40 lines, the same logic appears in two places, or a test can only pass by special-casing.
- Run commands with `uv run`.

## Amendment: id templates (approved after the plan was written)

`item.from_files`, `GeoRaster.to_items` and `GeoArray.to_items` take
`id: str | None = None`, a `GeoAnchor.format` template filled from each
Item's own raster (`scene.gs.anchor.format(id)`). None keeps the file-derived
id. `from_files` raises `ValueError` when two Items of one call get the same
id. Tests: `test_an_id_template_is_filled_per_scene` and
`test_an_id_template_that_repeats_refuses` in `tests/geodata/stac/test_item.py`
(Task 2); the accessors forward `id` (Task 4).

## Review Focus

1. A catalog read as a directory of part files must resolve relative hrefs against the directory, not its parent. Pinned in Task 3 (`test_a_directory_of_parts_reads_as_one_table`).
2. `to_items(paths)` given the wrong number of COG files must fail loudly, not pair files with the wrong scene. Pinned in Task 2 (`test_a_wrong_file_count_refuses`).
3. A plain vector with no time column must read back with exactly the columns it was written with. Pinned in Task 6 (`test_a_plain_vector_reads_back_with_its_own_columns`).
4. `table.load` on rows whose Parquet struct carries null entries for assets a row lacks must skip them, not crash on `None["href"]`. Pinned in Task 3 (`test_rows_with_different_assets_load`).
5. A stack with no dated group must refuse to build Items unless a time is given. Pinned in Task 2 (`test_a_timeless_stack_needs_a_time`).

## File Structure

| File | Responsibility after this plan |
| --- | --- |
| `src/geosave_engine/geodata/stac/asset.py` | One Asset from a raster and an href |
| `src/geosave_engine/geodata/stac/item.py` | Items from a raster, from a writer's files, from a stack |
| `src/geosave_engine/geodata/stac/table.py` (new) | Items to and from a GeoDataFrame and Parquet; rows to a raster |
| `src/geosave_engine/geodata/core/vector.py` | Spatial operations on any GeoDataFrame |
| `src/geosave_engine/geodata/core/array.py`, `raster.py`, `stack.py` | `to_items(paths)` delegates; `vectorize` on the array |
| `src/geosave_engine/geodata/io/geoparquet.py`, `geojson.py`, `geopackage.py`, `readers.py` | Plain vector formats |
| `src/geosave_engine/geodata/io/catalog.py` | Deleted |
| `src/geosave_engine/workflow/tasks/labels.py`, `src/geosave_engine/ml/segmentation/supervised/data.py` | Callers moved to `stac.item` and `stac.table` |

Shared test helper used by Tasks 1 to 4, placed in `tests/geodata/stac/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster


@pytest.fixture
def grid() -> GeoBox:
    return GeoBox.from_bbox(
        (300000, 5000000, 300640, 5000640), "EPSG:32633", resolution=10
    )


@pytest.fixture
def scene(grid):
    """Two dates of two uint16 bands with a fill value."""
    times = pd.to_datetime(["2025-06-01", "2025-06-11"])
    cube = np.arange(2 * 64 * 64, dtype="uint16").reshape(2, 64, 64)
    data = raster(
        {"red": (("time", "y", "x"), cube), "nir": (("time", "y", "x"), cube + 1)},
        grid,
        coords={"time": times},
    )
    for name in data.data_vars:
        data[name].attrs["_FillValue"] = np.uint16(0)
    return data


@pytest.fixture
def label(grid):
    """One timeless uint8 class plane on the same grid."""
    return raster({"label": (("y", "x"), np.ones((64, 64), dtype="uint8"))}, grid)
```

Append these fixtures to the existing `tests/geodata/stac/conftest.py`; keep its `local_source` fixture.

---

### Task 1: `asset.from_raster` names the media type from the href

**Files:**
- Modify: `src/geosave_engine/geodata/stac/asset.py`
- Modify: `tests/geodata/stac/conftest.py` (add the fixtures above)
- Test: `tests/geodata/stac/test_asset.py`

**Interfaces:**
- Produces: `asset.from_raster(raster: xr.Dataset, href: str | PathLike[str]) -> pystac.Asset`. `asset.from_path` and `asset.default_key` are unchanged in this task.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/stac/test_asset.py`:

```python
import pystac
import pytest

from geosave_engine.geodata.stac import asset


@pytest.mark.parametrize(
    ("name", "media_type"),
    [
        ("forest.zarr", pystac.MediaType.ZARR),
        ("forest.nc", pystac.MediaType.NETCDF),
        ("forest.tif", pystac.MediaType.GEOTIFF),
    ],
)
def test_the_media_type_follows_the_href_suffix(scene, tmp_path, name, media_type):
    built = asset.from_raster(scene, tmp_path / name)

    assert built.media_type == media_type
    assert built.href == str(tmp_path / name)
    assert built.roles == ["data"]


def test_an_href_that_names_no_raster_format_refuses(scene, tmp_path):
    with pytest.raises(ValueError, match="names no raster file or store"):
        asset.from_raster(scene, tmp_path / "forest")
```

In the same file, change every existing call `asset.from_raster(x, path, driver=...)` to `asset.from_raster(x, path)`. In `test_an_asset_built_at_write_time_agrees_with_the_file` compare every field except `type`, because a COG read back still reports the cloud-optimized profile while `from_raster` reports plain GeoTIFF.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/stac/test_asset.py -q`
Expected: FAIL with `TypeError: from_raster() missing 1 required keyword-only argument: 'driver'`

- [ ] **Step 3: Implement**

In `src/geosave_engine/geodata/stac/asset.py`:

Delete `_DRIVER_MEDIA_TYPES` and the `RasterDriver` import under `TYPE_CHECKING`. Rename `_SUFFIX_MEDIA_TYPES` to `_MEDIA_TYPES`. Replace `from_raster` with:

```python
def from_raster(raster: xr.Dataset, href: str | PathLike[str]) -> pystac.Asset:
    """Describe pixels saved at `href`, from the raster that holds them.

    No file is opened, so the raster has to be the one saved there.

    Args:
        raster: Raster the file or store holds.
        href: Local path or URL the raster is saved at.

    Returns:
        Asset with an absolute href, the media type its suffix names, the
        `data` role, its grid as projection fields, one raster and EO band per
        variable, and its first and last instant where the raster is dated.

    Raises:
        ValueError: The href names no raster file or store, or the raster
            carries no locatable grid.

    Examples:
        >>> from_raster(scene, "samples/forest.zarr").media_type
        'application/vnd+zarr'
    """
    location = absolute_location(href)
    media_type = _MEDIA_TYPES.get(PurePosixPath(location).suffix.lower())
    if media_type is None:
        raise ValueError(
            f"{href} names no raster file or store; pass a path a writer returned"
        )
    return _describe(raster, location, media_type)
```

In `from_path`, replace `_SUFFIX_MEDIA_TYPES` with `_MEDIA_TYPES`. Leave the rest of `from_path` and `_describe` untouched; `from_path` is deleted in Task 7.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/stac/test_asset.py -q`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `uv run pytest tests/geodata/stac -q && uv run ruff check src/geosave_engine/geodata/stac tests/geodata/stac`
Expected: PASS, no lint errors. Do not commit.

---

### Task 2: Items from a raster, from a writer's files, from a stack

**Files:**
- Modify: `src/geosave_engine/geodata/stac/item.py`
- Test: `tests/geodata/stac/test_item.py`

**Interfaces:**
- Consumes: `asset.from_raster(raster, href)`, `asset.default_key(raster)`.
- Produces:
  - `item.STACK_ID = "geosave:stack"`
  - `item.from_raster(raster: xr.Dataset, hrefs: Mapping[str, str | PathLike[str]], *, id: str, collection: str | None = None, datetime: DateTime | tuple[DateTime, DateTime] | None = None) -> pystac.Item`
  - `item.from_files(raster: xr.Dataset, paths: str | PathLike[str] | Sequence[str | PathLike[str]], *, collection: str | None = None, datetime=None) -> tuple[pystac.Item, ...]`
  - `item.from_stack(rasters: Mapping[str, xr.Dataset], paths: str | PathLike[str] | Mapping[str, Sequence[str | PathLike[str]]], *, name: str, datetime=None) -> tuple[pystac.Item, ...]`
  - `item.from_assets` still exists until Task 7.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/stac/test_item.py`:

```python
from datetime import UTC, datetime

import dask
import pytest

from geosave_engine.geodata import stack
from geosave_engine.geodata.stac import item


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


def test_a_store_is_one_item_with_no_collection(scene, tmp_path):
    path = scene.gs.to_zarr(tmp_path / "forest.zarr")

    (built,) = item.from_files(scene, path)

    assert built.id == "forest"
    assert built.collection_id is None
    assert list(built.assets) == ["image"]
    assert built.assets["image"].href == str(path)
    assert built.properties["start_datetime"] == "2025-06-01T00:00:00Z"
    built.validate()


def test_one_file_per_scene_is_one_item_per_instant(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest")

    built = item.from_files(scene, paths, collection="s2")

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert [each.collection_id for each in built] == ["s2", "s2"]
    assert [list(each.assets) for each in built] == [["image"], ["image"]]
    assert built[1].assets["image"].href == str(paths[1])


def test_split_bands_become_one_asset_per_variable(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest", split_bands=True)

    built = item.from_files(scene, paths)

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert list(built[0].assets) == ["red", "nir"]
    assert built[1].assets["nir"].href == str(paths[3])


def test_a_wrong_file_count_refuses(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest", split_bands=True)

    with pytest.raises(ValueError, match="3 files"):
        item.from_files(scene, paths[:3])


def test_building_items_computes_no_pixels_and_opens_no_file(
    scene, tmp_path, monkeypatch
):
    paths = scene.gs.to_cog(tmp_path / "forest")
    lazy = scene.chunk()
    monkeypatch.setattr("rasterio.open", _refuse)
    monkeypatch.setattr("xarray.open_dataset", _refuse)

    with dask.config.set(scheduler=_refuse):
        built = item.from_files(lazy, paths)

    assert len(built) == 2


def test_a_timeless_raster_needs_a_time(label, tmp_path):
    (path,) = label.gs.to_cog(tmp_path / "label")

    with pytest.raises(ValueError, match="pass datetime="):
        item.from_files(label, path)

    (built,) = item.from_files(label, path, datetime=datetime(2025, 6, 1, tzinfo=UTC))
    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_a_stack_of_cogs_is_one_item_per_group_scene(scene, label, tmp_path):
    tree = stack({"optical": scene, "label": label})
    paths = tree.gs.to_cog(tmp_path / "s0")

    built = item.from_stack(tree.gs.rasters, paths, name="s0")

    assert [each.id for each in built] == [
        "s0/optical_20250601T000000",
        "s0/optical_20250611T000000",
        "s0/label",
    ]
    assert [each.collection_id for each in built] == ["optical", "optical", "label"]
    assert {each.properties[item.STACK_ID] for each in built} == {"s0"}
    # The timeless label takes the span its dated sibling covers.
    assert built[2].properties["start_datetime"] == "2025-06-01T00:00:00Z"


def test_a_stack_store_points_each_item_at_its_group(scene, label, tmp_path):
    tree = stack({"optical": scene, "label": label})
    path = tree.gs.to_zarr(tmp_path / "s0.zarr")

    optical, classes = item.from_stack(tree.gs.rasters, path, name="s0")

    assert (optical.id, classes.id) == ("s0/optical", "s0/label")
    assert optical.assets["image"].href == str(path)
    assert optical.assets["image"].extra_fields["xarray:open_kwargs"] == {
        "group": "optical"
    }
    assert classes.assets["label"].extra_fields["xarray:open_kwargs"] == {
        "group": "label"
    }


def test_a_timeless_stack_needs_a_time(label, tmp_path):
    tree = stack({"label": label})
    paths = tree.gs.to_cog(tmp_path / "s0")

    with pytest.raises(ValueError, match="pass datetime="):
        item.from_stack(tree.gs.rasters, paths, name="s0")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/stac/test_item.py -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'from_files'`

- [ ] **Step 3: Implement**

In `src/geosave_engine/geodata/stac/item.py`, keep `from_assets` as it is and add the following. Imports to add: `from os import PathLike`, `from pathlib import PurePosixPath`, `import xarray as xr`, `from . import asset`.

```python
# Rows saved from one stack share this property, so they can be found again.
STACK_ID = "geosave:stack"

_TIFF_SUFFIXES = (".tif", ".tiff")

type ItemTime = DateTime | tuple[DateTime, DateTime]


def from_raster(
    raster: xr.Dataset,
    hrefs: Mapping[str, str | PathLike[str]],
    *,
    id: str,
    collection: str | None = None,
    datetime: ItemTime | None = None,
) -> pystac.Item:
    """Build one Item from a raster and where its pixels are saved.

    Args:
        raster: Raster the files hold. No pixel is read.
        hrefs: Asset names mapped to saved paths. One entry holds the whole
            raster; several are one variable each, named by the variable.
        id: Item identity.
        collection: STAC Collection ID. None states none.
        datetime: One instant, or a start and end. None takes the span the
            raster's time axis covers.

    Returns:
        Item over the raster's grid, dated by one instant or by a start and
        end, with one asset per href.

    Raises:
        KeyError: Several hrefs are given and one names no variable.
        ValueError: The raster has no locatable grid, an href names no raster
            format, or the raster is timeless and `datetime` is None.

    Examples:
        >>> from_raster(label, {"label": "s0/label.tif"}, id="s0").id
        's0'
    """
    assets = {}
    for key, href in hrefs.items():
        part = raster if len(hrefs) == 1 else raster[[key]]
        assets[key] = asset.from_raster(part, href)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    start, end = _span(assets, datetime)

    # STAC dates an Item by one instant, or by a start and an end, never both.
    ranged = start != end
    built = pystac.Item(
        id,
        mapping(footprint),
        list(footprint.bounds),
        None if ranged else start,
        {},
        start_datetime=start if ranged else None,
        end_datetime=end if ranged else None,
        collection=collection,
    )
    for extension in (ProjectionExtension, RasterExtension, EOExtension):
        extension.add_to(built)
    for key, entry in assets.items():
        built.add_asset(key, entry)
    return built


def _span(
    assets: Mapping[str, pystac.Asset], when: ItemTime | None
) -> tuple[DateTime, DateTime]:
    """Return the first and last instant an Item covers."""
    if isinstance(when, tuple):
        return when
    if when is not None:
        return when, when
    starts = []
    ends = []
    for entry in assets.values():
        times = entry.common_metadata
        if times.start_datetime is not None and times.end_datetime is not None:
            # PySTAC's getters truncate nanoseconds; read the strings it keeps.
            starts.append(pd.Timestamp(entry.extra_fields["start_datetime"]))
            ends.append(pd.Timestamp(entry.extra_fields["end_datetime"]))
    if not starts:
        raise ValueError(
            f"none of the assets {list(assets)} is dated, and a STAC Item needs a "
            f"time; pass datetime="
        )
    return min(starts), max(ends)


def from_files(
    raster: xr.Dataset,
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    collection: str | None = None,
    datetime: ItemTime | None = None,
) -> tuple[pystac.Item, ...]:
    """Build the Items of a raster from the paths its writer returned.

    Args:
        raster: Raster that was written. No pixel is read.
        paths: One store, or the COG files in time then variable order.
        collection: STAC Collection ID. None states none.
        datetime: Time for a timeless raster.

    Returns:
        One Item for a store, named after it. One Item per instant for COGs,
        named after its scene, with one asset per file.

    Raises:
        ValueError: No path is given, a store is given with other paths, or
            the COG count matches neither one file nor one file per variable
            for each instant.

    Examples:
        >>> [each.id for each in from_files(ds, ds.gs.to_cog("samples/forest"))]
        ['forest_20250601T103031', 'forest_20250611T103031']
    """
    files = [paths] if isinstance(paths, (str, PathLike)) else list(paths)
    if not files:
        raise ValueError("from_files needs the paths a writer returned")
    names = [PurePosixPath(str(file)) for file in files]

    # A store holds the whole raster, time axis included.
    if any(name.suffix.lower() not in _TIFF_SUFFIXES for name in names):
        if len(files) != 1:
            raise ValueError("a store is one path; pass COG files or one store")
        hrefs = {asset.default_key(raster): files[0]}
        store = from_raster(
            raster, hrefs, id=names[0].stem, collection=collection, datetime=datetime
        )
        return (store,)

    # COG files arrive one scene at a time: one file, or one per variable.
    dated = TIME_COORDINATE in raster.dims
    count = raster.sizes[TIME_COORDINATE] if dated else 1
    variables = raster.gs.variables
    width, extra = divmod(len(files), count)
    if extra or width not in (1, len(variables)):
        raise ValueError(
            f"{len(files)} files do not hold {count} scenes of {list(variables)}; "
            f"pass exactly the files to_cog returned"
        )
    items = []
    for index in range(count):
        scene = raster.isel({TIME_COORDINATE: index}) if dated else raster
        first = index * width
        if width == 1:
            scene_id = names[first].stem
            hrefs = {asset.default_key(scene): files[first]}
        else:
            # A split scene is a folder named after it, holding one file per band.
            scene_id = names[first].parent.name
            hrefs = dict(zip(variables, files[first : first + width], strict=True))
        items.append(
            from_raster(
                scene, hrefs, id=scene_id, collection=collection, datetime=datetime
            )
        )
    return tuple(items)


def from_stack(
    rasters: Mapping[str, xr.Dataset],
    paths: str | PathLike[str] | Mapping[str, Sequence[str | PathLike[str]]],
    *,
    name: str,
    datetime: ItemTime | None = None,
) -> tuple[pystac.Item, ...]:
    """Build the Items of every group of a saved stack.

    Args:
        rasters: Group names mapped to the rasters that were written.
        paths: What the stack writer returned: group names mapped to COG
            files, or one store holding every group.
        name: The stack's identity, shared by its Items.
        datetime: Time for timeless groups. None gives them the span the
            dated groups cover.

    Returns:
        Each group's Items in group order. A group's name is its collection,
        ids are prefixed with `name`, and every Item carries `geosave:stack`.

    Raises:
        ValueError: A group is timeless, no group is dated and `datetime` is
            None.

    Examples:
        >>> items = from_stack(sample.gs.rasters, sample.gs.to_cog("s0"), name="s0")
        >>> items[0].collection_id
        'optical'
    """
    spans = [raster.gs.timespan for raster in rasters.values()]
    covered = [span for span in spans if span is not None]
    fallback = datetime
    if fallback is None and covered:
        fallback = (min(span[0] for span in covered), max(span[1] for span in covered))

    items = []
    for group, raster in rasters.items():
        when = fallback if raster.gs.timespan is None else None
        if isinstance(paths, Mapping):
            built = from_files(raster, paths[group], collection=group, datetime=when)
            for each in built:
                each.id = f"{name}/{each.id}"
        else:
            key = asset.default_key(raster)
            whole = from_raster(
                raster,
                {key: paths},
                id=f"{name}/{group}",
                collection=group,
                datetime=when,
            )
            # The store holds every group; the reader needs this one's name.
            whole.assets[key].extra_fields["xarray:open_kwargs"] = {"group": group}
            built = (whole,)
        for each in built:
            each.properties[STACK_ID] = name
        items.extend(built)
    return tuple(items)
```

Add `from geosave_engine.geodata.conventions import TIME_COORDINATE` to the imports.

Drop `STACK_ID` from the existing `conventions` import, since the module now defines it; `from_assets` keeps importing `RASTER_ID`, `GROUP` and `GROUPS` until Task 7.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/stac/test_item.py -q`
Expected: PASS

If `test_a_stack_of_cogs_is_one_item_per_group_scene` fails on the label's `start_datetime` because `gs.timespan` returns timezone-naive datetimes that PySTAC serialises without `Z`, stop and report: the span type needs a decision, not a patch.

- [ ] **Step 5: Checkpoint**

Run: `uv run pytest tests/geodata/stac -q && uv run ruff check src/geosave_engine/geodata/stac tests/geodata/stac`
Expected: PASS. Do not commit.

---

### Task 3: The item table, and plain GeoParquet

The table takes over everything STAC-specific that `io/geoparquet.py` does
today, so both change together.

**Files:**
- Create: `src/geosave_engine/geodata/stac/table.py`
- Modify: `src/geosave_engine/geodata/stac/__init__.py`
- Modify: `src/geosave_engine/geodata/io/geoparquet.py` (`write`, href helpers)
- Modify: `src/geosave_engine/geodata/io/readers.py` (`read_vector`)
- Delete: `tests/geodata/core/test_catalog.py`, `tests/geodata/io/test_catalog.py`, `tests/geodata/io/test_catalog_records.py`
- Test: `tests/geodata/stac/test_table.py` (new), `tests/geodata/stac/test_native_items.py`, `tests/geodata/io/test_geoparquet.py`, `tests/geodata/core/test_vector.py`

**Interfaces:**
- Consumes: `item.from_files`, `item.from_stack`, `io.geoparquet.read`, `io.geoparquet.write`, `io.readers.read_raster`.
- Produces:
  - `table.from_items(items: Iterable[pystac.Item]) -> gpd.GeoDataFrame`
  - `table.write(items: Iterable[pystac.Item], path: str | PathLike[str], *, overwrite: bool = False, storage_options: StorageOptions | None = None) -> Path | str`
  - `table.read(path: str | PathLike[str], **options) -> gpd.GeoDataFrame`
  - `table.load(rows: gpd.GeoDataFrame, *, assets: str | Sequence[str] | None = None, **options) -> xr.Dataset`
  - `io.geoparquet.write` writes plain GeoParquet only; `read_vector` returns Parquet rows as stored.

- [ ] **Step 1: Write the failing tests**

Create `tests/geodata/stac/test_table.py`:

```python
import shutil

import dask
import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
import stac_geoparquet.arrow as stac_arrow

from geosave_engine.geodata import stack
from geosave_engine.geodata.stac import item, table


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


def _cog_items(scene, root, name="forest"):
    return item.from_files(scene, scene.gs.to_cog(root / name), collection="s2")


def test_items_become_rows_without_a_collection_column(scene, tmp_path):
    path = scene.gs.to_zarr(tmp_path / "forest.zarr")

    rows = table.from_items(item.from_files(scene, path))

    assert rows["id"].tolist() == ["forest"]
    assert "collection" not in rows
    assert "bbox" not in rows
    assert rows.crs.to_epsg() == 4326


def test_no_items_refuses():
    with pytest.raises(ValueError, match="at least one"):
        table.from_items([])


def test_a_written_table_is_stac_geoparquet(scene, tmp_path):
    items = _cog_items(scene, tmp_path)

    path = table.write(items, tmp_path / "catalog.parquet")

    read = list(stac_arrow.stac_table_to_items(pq.read_table(path)))
    assert [each["id"] for each in read] == [each.id for each in items]
    assert read[0]["bbox"] == pytest.approx(items[0].bbox)


def test_rows_load_back_as_the_raster_that_was_written(scene, tmp_path):
    table.write(_cog_items(scene, tmp_path), tmp_path / "catalog.parquet")
    rows = table.read(tmp_path / "catalog.parquet")

    with dask.config.set(scheduler=_refuse):
        loaded = table.load(rows)

    assert dict(loaded.sizes) == {"time": 2, "y": 64, "x": 64}
    np.testing.assert_array_equal(loaded["red"].values, scene["red"].values)
    loaded.close()


def test_a_table_follows_its_assets_when_moved(scene, tmp_path):
    home = tmp_path / "home"
    table.write(_cog_items(scene, home), home / "catalog.parquet")
    moved = tmp_path / "moved"
    shutil.move(home, moved)

    rows = table.read(moved / "catalog.parquet")

    href = rows.iloc[0]["assets"]["image"]["href"]
    assert href.startswith(str(moved))
    table.load(rows).close()


def test_a_directory_of_parts_reads_as_one_table(scene, tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    first, second = _cog_items(scene, tmp_path)
    table.write([first], catalog / "part-0.parquet")
    table.write([second], catalog / "part-1.parquet")

    rows = table.read(catalog)

    assert sorted(rows["id"]) == sorted([first.id, second.id])
    assert rows.iloc[0]["assets"]["image"]["href"].startswith(str(tmp_path / "forest"))
    table.load(rows).close()


def test_a_bbox_filter_reads_only_matching_rows(scene, tmp_path):
    table.write(_cog_items(scene, tmp_path), tmp_path / "catalog.parquet")

    inside = table.read(tmp_path / "catalog.parquet", bbox=(12.0, 45.0, 13.0, 46.0))
    outside = table.read(tmp_path / "catalog.parquet", bbox=(0.0, 0.0, 1.0, 1.0))

    assert len(inside) == 2
    assert len(outside) == 0


def test_rows_with_different_assets_load(scene, label, tmp_path):
    tree = stack({"optical": scene, "label": label})
    items = item.from_stack(tree.gs.rasters, tree.gs.to_cog(tmp_path / "s0"), name="s0")
    table.write(items, tmp_path / "catalog.parquet")
    rows = table.read(tmp_path / "catalog.parquet")

    classes = table.load(rows[rows["collection"] == "label"])

    assert list(classes.data_vars) == ["label"]
    classes.close()


def test_a_group_of_a_stack_store_loads_alone(scene, label, tmp_path):
    tree = stack({"optical": scene, "label": label})
    items = item.from_stack(
        tree.gs.rasters, tree.gs.to_zarr(tmp_path / "s0.zarr"), name="s0"
    )
    table.write(items, tmp_path / "catalog.parquet")
    rows = table.read(tmp_path / "catalog.parquet")

    optical = table.load(rows[rows["collection"] == "optical"])

    assert list(optical.data_vars) == ["red", "nir"]
    np.testing.assert_array_equal(optical["nir"].values, scene["nir"].values)
    optical.close()


def test_a_named_asset_no_row_has_is_a_key_error(scene, tmp_path):
    rows = table.from_items(_cog_items(scene, tmp_path))

    with pytest.raises(KeyError, match="mask"):
        table.load(rows, assets="mask")


def test_a_frame_without_assets_is_a_key_error():
    plain = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))

    with pytest.raises(KeyError, match="assets"):
        table.load(plain)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/stac/test_table.py -q`
Expected: FAIL with `ImportError: cannot import name 'table'`

- [ ] **Step 3: Implement**

Create `src/geosave_engine/geodata/stac/table.py`:

```python
"""Keep STAC Items as a table: in memory, on disk, and back to pixels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pystac

from geosave_engine.geodata.io import geoparquet
from geosave_engine.geodata.io.readers import read_raster
from geosave_engine.geodata.io.storage import absolute_location

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset, GeoDataFrame
    from geosave_engine.geodata.io.storage import StorageOptions


def from_items(items: Iterable[pystac.Item]) -> GeoDataFrame:
    """Build a table with one row per Item.

    Args:
        items: Items whose assets carry absolute hrefs, or a self href that
            resolves relative ones.

    Returns:
        GeoDataFrame in WGS84 with the Items' properties as columns and their
        assets as mappings. The Items are not changed.

    Raises:
        ValueError: No Item is given.
        pystac.STACError: A relative asset href has no Item self href.

    Examples:
        >>> from_items(ds.gs.to_items(paths))["id"].tolist()
        ['forest_20250601T103031', 'forest_20250611T103031']
    """
    from stac_geoparquet.arrow import parse_stac_items_to_arrow

    records = []
    for item in items:
        clone = item.clone()
        clone.make_asset_hrefs_absolute()
        records.append(clone.to_dict())
    if not records:
        raise ValueError("from_items needs at least one STAC Item")
    table = parse_stac_items_to_arrow(records).read_all()
    frame = gpd.GeoDataFrame.from_arrow(table)
    # The geometry states the bounds; the file's covering column is written from it.
    return cast("GeoDataFrame", frame.drop(columns="bbox"))


def write(
    items: Iterable[pystac.Item],
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> Path | str:
    """Write Items as one stac-geoparquet file.

    Hrefs on the table's own filesystem are stored relative to it, so a table
    and its assets move together. Add to a catalog by writing another file
    into its folder.

    Args:
        items: Items to store.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Local writes return a path; URL writes return the supplied URL.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: No Item is given, or the suffix is wrong.

    Examples:
        >>> write(ds.gs.to_items(paths), "data/catalog/part-0.parquet")
        PosixPath('data/catalog/part-0.parquet')
    """
    frame = _relative_hrefs(from_items(items), absolute_location(path))
    return geoparquet.write(
        frame,
        path,
        overwrite=overwrite,
        write_covering_bbox=True,
        storage_options=storage_options,
    )


def read(path: str | PathLike[str], **options: Any) -> GeoDataFrame:
    """Read an item table from one file or a folder of them.

    Args:
        path: Parquet file, or a folder of Parquet files, as a path or URL.
        **options: GeoParquet read options. `bbox`, `columns` and `filters`
            are applied while reading.

    Returns:
        GeoDataFrame whose asset hrefs open directly, each row holding only
        the assets it has.

    Examples:
        >>> rows = read("data/catalog", bbox=(12.0, 45.0, 13.0, 46.0))
        >>> rows.gs.query(anchor)["id"].tolist()
        ['forest_20250601T103031']
    """
    frame = geoparquet.read(path, **options)
    if "assets" not in frame:
        return cast("GeoDataFrame", frame)
    location = absolute_location(path)
    # A catalog folder holds part files whose hrefs are relative to the folder.
    folder = PurePosixPath(location).suffix == ""
    return cast("GeoDataFrame", _absolute_hrefs(frame, location, folder=folder))


def load(
    rows: gpd.GeoDataFrame,
    *,
    assets: str | Sequence[str] | None = None,
    **options: Any,
) -> Dataset:
    """Open the rasters selected rows point at as one lazy Dataset.

    Args:
        rows: Rows of an item table holding one raster: a store, or the
            scenes of one product.
        assets: Asset name or names to read. None reads every data asset.
        **options: Raster read options. Chunks default to an empty mapping.

    Returns:
        Lazy Dataset; scenes join along `time`.

    Raises:
        KeyError: The frame has no `assets`, or a named asset is in no row.
        ValueError: No row is selected, or the files do not form one raster.

    Examples:
        >>> load(rows[rows["collection"] == "optical"]).sizes["time"]
        2
    """
    if "assets" not in rows:
        raise KeyError("assets")
    wanted = [assets] if isinstance(assets, str) else assets
    hrefs = []
    found = set()
    opening: dict[str, Any] = {}
    for cell in rows["assets"]:
        for key, entry in cell.items():
            # Parquet holds one struct for every row, null where a row has none.
            if entry is None or (wanted is not None and key not in wanted):
                continue
            roles = entry.get("roles")
            if roles is not None and "data" not in roles:
                continue
            hrefs.append(str(entry["href"]))
            found.add(key)
            opening.update(_open_options(entry))
    if wanted is not None:
        for key in wanted:
            if key not in found:
                raise KeyError(key)
    if not hrefs:
        raise ValueError("load needs at least one row with a data asset")
    source = hrefs[0] if len(hrefs) == 1 else hrefs
    return read_raster(source, **{"chunks": {}, **opening, **options})


def _open_options(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reader options an asset states, without Parquet's null fills."""
    stated = entry.get("xarray:open_kwargs") or {}
    return {name: value for name, value in stated.items() if value is not None}
```

Add the two href helpers to the same file. They are the ones in
`io/geoparquet.py` today, made private, with a folder base for part files:

```python
def _relative_hrefs(frame: gpd.GeoDataFrame, table: str) -> gpd.GeoDataFrame:
    """Rewrite asset hrefs relative to the table that stores them."""
    assets = []
    for cell in frame["assets"]:
        row = {}
        for key, entry in cell.items():
            if entry is None:
                continue
            href = make_relative_href(str(entry["href"]), table)
            row[key] = {**entry, "href": href}
        assets.append(row)
    return cast("gpd.GeoDataFrame", frame.assign(assets=assets))


def _absolute_hrefs(
    frame: gpd.GeoDataFrame, table: str, *, folder: bool
) -> gpd.GeoDataFrame:
    """Rewrite stored hrefs into directly openable ones, dropping null assets."""
    assets = []
    for cell in frame["assets"]:
        row = {}
        for key, entry in cell.items():
            # Parquet holds one struct for every row, null where a row has none.
            if entry is None:
                continue
            fields = {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in entry.items()
                if value is not None
            }
            # A URL or a rooted path names its file already.
            href = str(fields["href"])
            protocol, _ = split_protocol(href)
            if protocol is None and not href.startswith("/"):
                href = make_absolute_href(href, table, start_is_dir=folder)
            row[key] = {**fields, "href": href}
        assets.append(row)
    return cast("gpd.GeoDataFrame", frame.assign(assets=assets))
```

Give `table.py` these imports in addition to the ones above: `import numpy as np`,
`from pathlib import PurePosixPath`, `from fsspec.core import split_protocol`,
`from pystac.utils import make_absolute_href, make_relative_href`.

In `src/geosave_engine/geodata/stac/__init__.py`, import `table` beside `asset` and `item` and add `"table"` to `__all__`.

`src/geosave_engine/geodata/io/geoparquet.py`:

- Delete `relative_hrefs` and `absolute_hrefs`.
- In `write`, delete the `frame = relative_hrefs(...)` block and the stac-geoparquet branch of the inner `save`, so `save` is only:

```python
    def save(destination: str | Path, **location: Any) -> None:
        gdf.to_parquet(destination, **location, **write_options, **parquet_options)
```

- In the `write` docstring, delete the paragraph about STAC tables.
- Remove the imports only the deleted code used (`to_parquet`, `split_protocol`, `make_absolute_href`, `make_relative_href`, `Mapping`, `np`, `absolute_location`); confirm each with ruff.
- Leave `read` and its `datetime_column` for Task 6.

`src/geosave_engine/geodata/io/readers.py`, `read_vector`: replace the Parquet branch with

```python
    if suffix in (".parquet", ".geoparquet"):
        return cast("GeoDataFrame", geoparquet.read(source, **options))
```

and rewrite the docstring's second paragraph, keeping only the `plantations.geojson` example:

```python
    GeoParquet accepts local paths or fsspec URLs; GeoJSON and GeoPackage are
    local only. An item table is read with `stac.table.read`, which also makes
    its asset hrefs openable.
```

- [ ] **Step 4: Remove the tests of the behaviour that moved**

```bash
rm tests/geodata/core/test_catalog.py tests/geodata/io/test_catalog.py tests/geodata/io/test_catalog_records.py
```

Those files test `catalog.write` followed by `read_vector(...).gs.to_raster()`, which relied on `read_vector` rewriting hrefs. Their round trips are covered by `test_table.py` and `test_item.py`. The remote-bucket cases in `tests/geodata/core/test_catalog.py` (`test_the_loop_runs_on_a_bucket` and the Hugging Face ones) have no replacement yet; list them in the final report.

`tests/geodata/core/test_vector.py`: delete `test_asset_href_round_trips_relative_to_a_movable_manifest`, `test_absolute_asset_outside_the_manifest_remains_absolute`, `test_relative_asset_expands_against_the_manifest`, `test_a_stac_table_carries_a_bounding_box_and_its_format_key`, `test_stac_tools_read_the_table`, `test_a_row_reads_back_only_the_assets_it_has`, `test_an_opaque_asset_pointer_fails_before_writing`, and the helpers `_record` and `_href` once nothing uses them.

`tests/geodata/io/test_geoparquet.py`: delete `test_remote_reference_table_round_trip_filters_and_materializes_hrefs`, `test_external_remote_asset_does_not_require_its_driver` and `test_opaque_asset_pointer_fails_before_writing`. `test_a_row_without_assets_stays_without` must still pass.

`tests/geodata/stac/test_native_items.py`: move its calls onto the table API. `frame.gs.to_geoparquet(path)` on an item table becomes `table.write(items, path)`, and `read_vector(path)` on one becomes `table.read(path)`. Delete `test_an_edited_geometry_changes_the_bbox_written`, which edits a frame and rewrites it as an item table; that path no longer exists.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/stac tests/geodata/io/test_geoparquet.py tests/geodata/io/test_readers.py -q`
Expected: PASS

- [ ] **Step 6: Checkpoint**

Run: `uv run pytest tests/geodata tests/workflow tests/ml -q -x && uv run ruff check src/geosave_engine/geodata tests/geodata`
Expected: PASS. A failure in a test not named above means something else depended on `read_vector` rewriting hrefs: stop and report it. Do not commit.

---

### Task 4: `to_items(paths)` and `vectorize` on the raster accessors

**Files:**
- Modify: `src/geosave_engine/geodata/core/raster.py` (the `to_items` overloads and method)
- Modify: `src/geosave_engine/geodata/core/stack.py` (`to_items`)
- Modify: `src/geosave_engine/geodata/core/array.py` (`to_items`, add `vectorize`)
- Test: `tests/geodata/core/test_raster.py`, `tests/geodata/core/test_stack.py`, `tests/geodata/transform/test_vector.py`

**Interfaces:**
- Consumes: `item.from_files`, `item.from_stack`.
- Produces:
  - `GeoRaster.to_items(paths=None, *, collection=None, datetime=None) -> tuple[pystac.Item, ...]`
  - `GeoStack.to_items(paths, *, name, datetime=None) -> tuple[pystac.Item, ...]`
  - `GeoArray.to_items(paths, *, collection=None, datetime=None) -> tuple[pystac.Item, ...]`
  - `GeoArray.vectorize(*, value_name="value", mask=None, connectivity=4) -> GeoDataFrame`

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/core/test_raster.py` (reuse that file's existing raster-building helper; the names `built` and `tmp_path` below stand for a dated two-variable raster and pytest's fixture):

```python
def test_to_items_describes_the_files_a_writer_returned(tmp_path):
    source = _dated_raster()
    paths = source.gs.to_cog(tmp_path / "forest")

    items = source.gs.to_items(paths, collection="s2")

    assert len(items) == source.sizes["time"]
    assert items[0].assets["image"].href == str(paths[0])
    assert items[0].collection_id == "s2"


def test_to_items_without_paths_describes_where_a_raster_was_read_from(tmp_path):
    source = _dated_raster()
    path = source.gs.to_zarr(tmp_path / "forest.zarr")

    with read_raster(path) as saved:
        (item,) = saved.gs.to_items()

    assert item.id == "forest"
    assert item.assets["image"].href == str(path)


def test_to_items_without_paths_refuses_an_unsaved_raster():
    with pytest.raises(ValueError, match="not saved"):
        _dated_raster().gs.to_items()


def test_a_band_indexes_as_its_one_variable_raster(tmp_path):
    source = _dated_raster()
    band = source["red"]
    paths = band.gs.to_cog(tmp_path / "red")

    items = band.gs.to_items(paths)

    assert [list(item.assets) for item in items] == [["red"]] * source.sizes["time"]
```

If `test_raster.py` has no helper returning a dated raster with a `red` variable, add this one at the top of the file:

```python
def _dated_raster():
    grid = GeoBox.from_bbox((300000, 5000000, 300160, 5000160), "EPSG:32633", resolution=10)
    cube = np.arange(2 * 16 * 16, dtype="uint16").reshape(2, 16, 16)
    return raster(
        {"red": (("time", "y", "x"), cube), "nir": (("time", "y", "x"), cube + 1)},
        grid,
        coords={"time": pd.to_datetime(["2025-06-01", "2025-06-11"])},
    )
```

In `tests/geodata/core/test_stack.py`, replace the two tests that call `sample.gs.to_items(tmp_path / "s0", ...)` and `GeoVector.from_items(...).gs.to_stack()` (around lines 285 to 325) with:

```python
def test_a_saved_stack_indexes_one_item_per_group(tmp_path):
    sample = _sample()
    paths = sample.gs.to_cog(tmp_path / "s0")

    items = sample.gs.to_items(paths, name="s0")

    assert {item.collection_id for item in items} == set(sample.gs.groups)
    assert {item.properties["geosave:stack"] for item in items} == {"s0"}


def test_a_stack_store_restores_each_group_through_the_table(tmp_path):
    from geosave_engine.geodata.stac import table

    sample = _sample()
    items = sample.gs.to_items(sample.gs.to_zarr(tmp_path / "s0.zarr"), name="s0")
    rows = table.from_items(items)

    restored = build_stack(
        {
            group: table.load(rows[rows["collection"] == group])
            for group in sample.gs.groups
        }
    )

    assert restored.gs.groups == sample.gs.groups
    restored.close()
```

`_sample()` stands for whatever stack those two tests built; keep their existing construction code and name it `_sample` if it was inline.

In `tests/geodata/transform/test_vector.py`, change the three `GeoVector.vectorize(x, ...)` calls (lines 208, 219, 228) to `x.gs.vectorize(...)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_raster.py tests/geodata/core/test_stack.py tests/geodata/transform/test_vector.py -q`
Expected: FAIL: `to_items` tries to write to the tuple of paths, and `'GeoArray' object has no attribute 'vectorize'`.

- [ ] **Step 3: Implement**

`src/geosave_engine/geodata/core/raster.py`: delete the four `@overload` definitions of `to_items` and replace the method with:

```python
    def to_items(
        self,
        paths: str | PathLike[str] | Sequence[str | PathLike[str]] | None = None,
        *,
        collection: str | None = None,
        datetime: ItemTime | None = None,
    ) -> tuple[pystac.Item, ...]:
        """Describe this raster's saved files as STAC Items.

        Nothing is written and no file is opened: the Items state this
        raster's grid, bands and time, and point at `paths`.

        Args:
            paths: What `to_cog`, `to_zarr` or `to_netcdf` returned. None uses
                the one file or store this raster was read from.
            collection: STAC Collection ID. None states none.
            datetime: Time for a timeless raster, as an instant or a start
                and end.

        Returns:
            One Item for a store, or one per instant for COGs.

        Raises:
            ValueError: `paths` is None and this raster was not read from one
                file or store; the files do not match this raster's scenes;
                or the raster is timeless and `datetime` is None.

        Examples:
            >>> items = ds.gs.to_items(ds.gs.to_cog("samples/forest"))
            >>> [item.id for item in items]
            ['forest_20250601T103031', 'forest_20250611T103031']
            >>> read_raster("samples/forest.zarr").gs.to_items()[0].id
            'forest'
        """
        from geosave_engine.geodata.stac import item

        if paths is None:
            paths = self._data.encoding.get("source")
            if paths is None:
                raise ValueError(
                    "this raster is not saved; pass the paths its writer returned"
                )
        return item.from_files(
            self._data, paths, collection=collection, datetime=datetime
        )
```

Under `TYPE_CHECKING` add `from collections.abc import Sequence` (if absent) and `from geosave_engine.geodata.stac.item import ItemTime`. Remove imports the deleted overloads were the only users of (`overload`, `Literal`, `RasterDriver`, the per-format option types) after checking each with `grep -n "<name>" src/geosave_engine/geodata/core/raster.py`.

`src/geosave_engine/geodata/core/stack.py`: replace `to_items` with:

```python
    def to_items(
        self,
        paths: str | PathLike[str] | Mapping[str, Sequence[str | PathLike[str]]],
        *,
        name: str,
        datetime: ItemTime | None = None,
    ) -> tuple[pystac.Item, ...]:
        """Describe this stack's saved groups as STAC Items.

        Nothing is written and no file is opened.

        Args:
            paths: What `to_cog`, `to_zarr` or `to_netcdf` returned.
            name: The stack's identity. Its Items share it as `geosave:stack`
                and carry it as their id prefix.
            datetime: Time for timeless groups. None gives them the span the
                dated groups cover.

        Returns:
            Each group's Items, with the group's name as their collection.

        Raises:
            ValueError: No group is dated and `datetime` is None.

        Examples:
            >>> items = sample.gs.to_items(sample.gs.to_cog("samples/s0"), name="s0")
            >>> sorted({item.collection_id for item in items})
            ['label', 'optical']
        """
        from geosave_engine.geodata.stac import item

        return item.from_stack(self.rasters, paths, name=name, datetime=datetime)
```

Add `Sequence` to the `collections.abc` import and `ItemTime` under `TYPE_CHECKING`; drop `RasterDriver` if unused.

`src/geosave_engine/geodata/core/array.py`: replace `to_items` and add `vectorize` after it:

```python
    def to_items(
        self,
        paths: str | PathLike[str] | Sequence[str | PathLike[str]],
        *,
        collection: str | None = None,
        datetime: ItemTime | None = None,
    ) -> tuple[pystac.Item, ...]:
        """Describe this band's saved files as STAC Items.

        Args:
            paths: What `to_cog` returned for this band.
            collection: STAC Collection ID. None states none.
            datetime: Time for a timeless band.

        Returns:
            One Item per instant, each holding this band as its one asset.

        Raises:
            ValueError: This band is unnamed, or timeless without `datetime`.

        Examples:
            >>> band = ds["ndvi"]
            >>> [item.id for item in band.gs.to_items(band.gs.to_cog("samples/ndvi"))]
            ['ndvi_20250601T103031', 'ndvi_20250611T103031']
        """
        return self.to_raster().gs.to_items(
            paths, collection=collection, datetime=datetime
        )

    def vectorize(
        self,
        *,
        value_name: str = "value",
        mask: xr.DataArray | np.ndarray | None = None,
        connectivity: Literal[4, 8] = 4,
    ) -> GeoDataFrame:
        """Polygonize contiguous values of this flag plane.

        This computes lazy flags, because geometry depends on their values.

        Args:
            value_name: Property column receiving each region's value.
            mask: Optional exact-grid mask selecting additional valid pixels.
            connectivity: Four- or eight-neighbour region connectivity.

        Returns:
            One row per contiguous region, in this array's CRS.

        Raises:
            ValueError: The array, mask, name, connectivity or dtype is
                unsuitable for polygonization.

        Examples:
            >>> prediction.gs.vectorize(value_name="class")["class"].tolist()
            [1, 2]
        """
        from geosave_engine.geodata.transform.vector import vectorize

        return vectorize(
            self._data, value_name=value_name, mask=mask, connectivity=connectivity
        )
```

Add any of `Literal`, `Sequence`, `GeoDataFrame`, `ItemTime` that `array.py` does not import yet.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/core/test_raster.py tests/geodata/core/test_stack.py tests/geodata/transform/test_vector.py tests/geodata/stac -q`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `uv run pytest tests/geodata -q -x`
Expected: failures only in `tests/geodata/core/test_vector.py` and `tests/geodata/io/test_geojson.py` tests that call `gs.to_raster`, `gs.to_stack` or `GeoVector.vectorize` on rows produced by the old `to_items(path)`. List them; they are rewritten or removed in Task 6. Any other failure: stop and report. Do not commit.

---

### Task 5: Callers move to `stac.item` and `stac.table`

**Files:**
- Modify: `src/geosave_engine/workflow/tasks/labels.py`
- Modify: `src/geosave_engine/ml/segmentation/supervised/data.py`
- Test: `tests/workflow/tasks/test_labels.py`, `tests/ml/segmentation/supervised/test_data.py`, `tests/ml/segmentation/supervised/test_module.py`, `tests/cli/core/test_workspace.py`

**Interfaces:**
- Consumes: `item.from_raster`, `item.STACK_ID`, `table.from_items`, `table.read`, `table.write`, `table.load`, `GeoStack.to_items(paths, name=)`.
- Produces: `supervised.Dataset(manifest, spec)` where `manifest` is an item table with one row per group and a `geosave:stack` column.

- [ ] **Step 1: Update the tests**

`tests/workflow/tasks/test_labels.py`: no assertion changes are expected. Run it first to see what the current behaviour is.

`tests/ml/segmentation/supervised/test_data.py`: replace the body of `_manifest` from the `folder = root / f"s{index}"` line to its `return` with:

```python
        groups = {"optical": optical}
        if label or index == 0:
            groups["label"] = classes
        sample = stack(groups)
        if times:
            paths = sample.gs.to_zarr(root / f"s{index}.zarr")
            when = None
        else:
            paths = sample.gs.to_cog(root / f"s{index}")
            when = datetime(2025, 6, 1, tzinfo=UTC)
        items.extend(sample.gs.to_items(paths, name=f"s{index}", datetime=when))
    path = table.write(items, root / "manifest.parquet")
    # Building read each sample lazily; collect, so no test starts with them open.
    gc.collect()
    return table.read(path)
```

Initialise `items = []` where `records = []` was, rename the label variable from `class` to `label` in the `classes` raster so the group and its variable agree with the target, and update imports: add `from geosave_engine.geodata import stack` and `from geosave_engine.geodata.stac import table`; drop `pystac`, `GeoVector`, `read_vector` and `io` if nothing else in the file uses them.

Then adapt the assertions that depended on one row per sample:

- Line 512, `manifest["id"] = "duplicate"`: keep; the Dataset still refuses duplicate Item ids.
- Line 524, `manifest["class_id"] = [7, 9]`: change to `manifest["class_id"] = manifest["geosave:stack"].map({"s0": 7, "s1": 9})`.
- Lines 530 to 535: delete the `source_assets` assertion, the `assert "assets" not in row` line and the `pytest.raises(KeyError, match="assets")` block around `gs.to_stack`.
- Any test matching the missing-asset message (`has no ['label'] asset`): change the expected text to `has no ['label'] group`.

`tests/ml/segmentation/supervised/test_module.py` line 654 reuses `_manifest`; no change beyond the import.

`tests/cli/core/test_workspace.py` lines 118 to 135: replace the `entries = [...]` construction and `samples = GeoVector.from_items(entries)` with:

```python
    from geosave_engine.geodata import read_raster, stack
    from geosave_engine.geodata.stac import table

    entries = []
    for path in sorted(examples.iterdir()):
        if not path.is_dir():
            continue
        # Each sample folder holds one raster per layer, named after it.
        layers = sorted(path.iterdir())
        sample = stack({layer.stem: read_raster(layer) for layer in layers})
        files = {layer.stem: (layer,) for layer in layers}
        entries.extend(sample.gs.to_items(files, name=path.name))
    for split in ("train", "val"):
        (workspace / f"data/{split}").mkdir(parents=True, exist_ok=True)
        table.write(
            entries, workspace / f"data/{split}/manifest.parquet", overwrite=True
        )
```

and delete the old `samples.gs.to_geoparquet(...)` loop and the now-unused `asset`, `item`, `GeoVector` imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/ml/segmentation/supervised/test_data.py -q -x`
Expected: FAIL in `Dataset.__init__`, which still looks for every group among one row's assets.

- [ ] **Step 3: Implement**

`src/geosave_engine/workflow/tasks/labels.py`: change the imports to

```python
from geosave_engine.geodata import GeoDataFrame, read_raster
from geosave_engine.geodata.stac import item, table
```

Replace the directory branch of `read_labels` with:

```python
    path = Path(source)
    if path.suffix.lower() not in _TABLE_SUFFIXES:
        rows = []
        for sample_id, label_path in find_labels(path, pattern).items():
            with read_raster(label_path) as label:
                if label.gs.timespan is None:
                    raise ValueError(f"Label raster has no time: {label_path}")
                rows.append(
                    item.from_raster(label, {"label": label_path}, id=sample_id)
                )
        return table.from_items(rows)

    labels = table.read(path)
```

and keep the validation below it unchanged.

`src/geosave_engine/ml/segmentation/supervised/data.py`:

Change the imports: replace `from geosave_engine.geodata import read_vector, stack` with `from geosave_engine.geodata import stack`, and add

```python
from geosave_engine.geodata.stac import table
from geosave_engine.geodata.stac.item import STACK_ID
```

In `Dataset.__init__`, replace the block from `self._groups = (*spec.rasters, target)` through `self._manifest = manifest` with:

```python
        self._groups = (*spec.rasters, target)
        absent = [name for name in (STACK_ID, "collection", "assets") if name not in manifest]
        if absent:
            raise ValueError(
                f"the manifest has no {absent} column; build it from each sample "
                f"stack's to_items"
            )
        if not manifest["id"].is_unique:
            raise ValueError("manifest IDs must be unique non-empty strings")

        # A sample is the rows saved from one stack, one group per collection.
        self._samples = {
            str(name): rows for name, rows in manifest.groupby(STACK_ID, sort=False)
        }
        for sample_id, rows in self._samples.items():
            present = set(rows["collection"])
            missing = [name for name in self._groups if name not in present]
            if missing:
                raise ValueError(
                    f"sample {sample_id!r} has no {missing} group; its groups are "
                    f"{sorted(present)}"
                )
```

In `_make_reference`, replace the three lines that set `source_id` and `source_assets` with:

```python
        rows = [self._source_rows[key] for key in reference.parent_id]
        reference["source_id"] = [row[STACK_ID] for row in rows]
```

and leave the annotation-copying block below it as it is.

In `_prepare`, replace the loop header and the `sample = ...` statement:

```python
        for source_id, rows in self._samples.items():
            groups = {
                name: rows[rows["collection"] == name] for name in self._groups
            }
            sample = stack({name: table.load(part) for name, part in groups.items()})
            # The target's row is the sample's record: it carries the annotations.
            row = groups[self.target].iloc[0]
```

Delete the old `row = self._manifest.iloc[position]`, `source_id = row["id"]` and `gs.to_stack(...)` lines. `self._source_rows[parent_id] = row.to_dict()` stays.

In the DataModule's split reader (around line 278), replace `read_vector(manifest)` with `table.read(manifest)`.

Update the `Dataset` docstring's `manifest` line to: `manifest: Item table with one row per group; rows sharing `geosave:stack` form one sample.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/workflow/tasks/test_labels.py tests/ml/segmentation/supervised -q`
Expected: PASS

If a Dataset test fails because `spec.preprocess` or `spec.frames.cut` receives groups in a different shape than before, stop and report: that is the deferred sample-manifest design surfacing, not something to patch here.

- [ ] **Step 5: Checkpoint**

Run: `uv run pytest tests/workflow tests/ml tests/model -q && uv run ruff check src/geosave_engine/workflow src/geosave_engine/ml`
Expected: PASS. `tests/cli/core/test_workspace.py::test_segmentation_workspace_trains_on_prepared_samples` is marked slow; run it with `uv run pytest tests/cli/core/test_workspace.py -m slow -q` and report the result either way. Do not commit.

---

### Task 6: GeoVector and vector I/O become plain

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `src/geosave_engine/geodata/io/geoparquet.py` (`read`), `geojson.py`, `geopackage.py`
- Test: `tests/geodata/core/test_vector.py`, `tests/geodata/io/test_geojson.py`, `tests/geodata/io/test_geoparquet.py`, `tests/geodata/io/test_readers.py`, `tests/geodata/stac/test_native_items.py`, `tests/geodata/stac/test_table.py`

**Interfaces:**
- Produces: `GeoVector` with exactly `crs`, `footprint`, `query`, `rasterize`, `to_geojson`, `to_geopackage`, `to_geoparquet`, `concat`, `from_geometry`. Vector readers take no `datetime_column`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/core/test_vector.py`:

```python
REMOVED = (
    "normalize", "timespan", "to_anchor", "to_items", "from_items", "upsert",
    "to_raster", "to_stack", "raster_ids", "stack_ids", "empty", "vectorize",
)


@pytest.mark.parametrize("name", REMOVED)
def test_geovector_is_spatial_only(name: str) -> None:
    assert not hasattr(GeoVector, name)


def test_a_geometry_factory_adds_no_time_column() -> None:
    frame = GeoVector.from_geometry(Point(0, 0), properties={"name": "a"})

    assert list(frame.columns) == ["name", "geometry"]
```

Add to `tests/geodata/io/test_readers.py`:

```python
@pytest.mark.parametrize("suffix", [".geojson", ".gpkg", ".parquet"])
def test_a_plain_vector_reads_back_with_its_own_columns(tmp_path, suffix):
    plots = gpd.GeoDataFrame(
        {"class": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )
    path = tmp_path / f"plots{suffix}"
    writer = {
        ".geojson": plots.gs.to_geojson,
        ".gpkg": plots.gs.to_geopackage,
        ".parquet": plots.gs.to_geoparquet,
    }[suffix]

    read = read_vector(writer(path))

    assert sorted(read.columns) == ["class", "geometry"]
```

(Add `import geopandas as gpd`, `from shapely.geometry import Point` and `read_vector` to that file's imports if absent.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_vector.py -k "spatial_only or no_time_column" tests/geodata/io/test_readers.py -k plain_vector -q`
Expected: FAIL: the removed names still exist and `datetime` is among the columns.

- [ ] **Step 3: Implement the plain I/O**

`src/geosave_engine/geodata/io/geoparquet.py`, `read`: delete the `datetime_column` parameter and its docstring line and the `GeoVector` import, and return the frame from `gpd.read_parquet` directly.

`src/geosave_engine/geodata/io/geojson.py`:

- `read`: delete `datetime_column`, the `payload`/`is_stac` detection and the whole `if is_stac:` branch, leaving:

```python
    driver_options = dict(open_options.pop("driver_options", {}))
    if "engine" in driver_options:
        raise ValueError("driver_options must not contain engine; use engine=")
    if "ignore_geometry" in driver_options:
        raise ValueError("driver_options must not contain ignore_geometry")
    options: dict[str, Any] = {**open_options, **driver_options}
    frame = gpd.read_file(source, engine=engine, ignore_geometry=False, **options)
    return cast("gpd.GeoDataFrame", frame)
```

- `write`: delete the `if "stac_version" in gdf:` branch.
- Fix both docstrings (no STAC sentences) and remove unused imports (`json`, `pystac`, `pd`, `box`, `absolute_location`, `filesystem_path`).

`src/geosave_engine/geodata/io/geopackage.py`: delete `datetime_column`, its docstring line and the `GeoVector` import; `return frame`.

- [ ] **Step 4: Trim GeoVector**

In `src/geosave_engine/geodata/core/vector.py`:

- Delete the methods `normalize`, `timespan`, `to_anchor`, `raster_ids`, `stack_ids`, `to_raster`, `to_stack`, `to_items`, `from_items`, `upsert`, `vectorize`, `empty`.
- Replace `_time_bounds` with:

```python
    def _time_bounds(self) -> tuple[pd.Series, pd.Series] | None:
        """Return each row's first and last instant in UTC, or None if undated.

        A frame is dated when it has `datetime`, `start_datetime` or
        `end_datetime`. A row missing an edge leaves it NaT, so a time query
        treats that edge as open.
        """
        frame = self._data
        times = {
            name: pd.to_datetime(frame[name], utc=True, format="mixed")
            for name in ("datetime", "start_datetime", "end_datetime")
            if name in frame
        }
        if not times:
            return None
        instant = times.get(
            "datetime",
            pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"),
        )
        first = times["start_datetime"].fillna(instant) if "start_datetime" in times else instant
        last = times["end_datetime"].fillna(instant) if "end_datetime" in times else instant
        return first, last
```

- In `query`, replace the time block with:

```python
        span = anchor.timespan
        bounds = matched.gs._time_bounds()
        if span is not None and bounds is not None:
            start, end = (pd.Timestamp(naive_utc(edge), tz="UTC") for edge in span)
            first, last = bounds
            matched = matched.loc[
                (first.isna() | (first <= end)) & (last.isna() | (last >= start))
            ]
```

- In `from_geometry`, return the frame directly instead of through `cls.normalize(...)`:

```python
        frame = gpd.GeoDataFrame(
            {name: [value] for name, value in properties.items()},
            geometry=[to_shapely(geometry)],
            crs=geometry_crs or crs or "EPSG:4326",
        )
        return cast("GeoDataFrame", frame)
```

- Rewrite the class docstring:

```python
    """GeoSave operations on a GeoDataFrame, read through `gs`.

    A vector is geometry with information attached: an active geometry column,
    a CRS, and any other columns. Nothing here requires or adds a column. A
    time filter applies in `query` only where the frame happens to carry
    `datetime`, `start_datetime` or `end_datetime`.

    Args:
        data: GeoDataFrame with an active geometry column.

    Raises:
        AttributeError: `data` is not a GeoDataFrame, or has no active
            geometry column.

    Examples:
        >>> plots = read_vector("plots.geojson")
        >>> plots.gs.crs.to_epsg()
        4326
        >>> mask = plots.gs.rasterize(scene, column="class")
        >>> plots.gs.to_geoparquet("plots.parquet")
        PosixPath('plots.parquet')
    """
```

- In `to_geoparquet`'s docstring, delete the paragraph about STAC tables and the `catalog` example.
- Remove unused imports: `Iterable` stays for `concat`; drop `pystac`, `Sequence` if unused, `AnchorDatetime`, `DateRange`, `GridAnchor`, `Any` if unused. Check each with `uv run ruff check src/geosave_engine/geodata/core/vector.py`.

- [ ] **Step 5: Bring the existing tests in line**

`tests/geodata/core/test_vector.py`, delete these tests (they test removed behaviour): `test_empty_vector_has_a_crs_and_no_footprint`, `test_upsert_replaces_caller_key_and_appends_new_key`, `test_upsert_rejects_invalid_incoming_identity`, `test_geometry_factory_normalizes_stac_time_without_inventing_it`, `test_normalize_maps_an_explicit_date_column_without_mutating_source`, `test_selected_vector_builds_anchor_with_its_temporal_bounds`, `test_plain_vector_has_no_raster_or_stack_identities`, `test_empty_vector_retains_normalized_missing_time`, and the helper `_keyed` once nothing uses it.

In the same file:
- The `vectorize` tests (`test_vectorize_*`, `test_vector_method_delegates_to_vector_transform`): change `GeoVector.vectorize(x, ...)` to `x.gs.vectorize(...)`.
- `test_query_empty_collection_preserves_schema`: build the empty frame with `gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))`.
- `test_undated_records_remain_spatially_selectable_in_a_time_query`: keep; if it built its frame through `normalize`, build it with `gpd.GeoDataFrame` and an explicit `datetime` column holding `pd.NaT`.

`tests/geodata/transform/test_vector.py` line 104: replace `GeoVector.empty("EPSG:32633")` with `gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:32633"))`.

`tests/geodata/io/test_geojson.py`: delete every test that converts between frames and Items or reads STAC JSON (`test_item_conversion_preserves_native_fields_and_json`, `test_geometry_records_become_items_after_explicit_id_and_time`, `test_stac_json_round_trip_keeps_fields_at_the_top_level`, `test_stac_item_json_reader_resolves_relative_assets`, `test_geojson_reader_maps_a_named_property_to_canonical_time`, `test_direct_readers_normalize_undated_geometry`, `test_stac_json_reader_filters_records_without_flattening_assets`, `test_range_records_export_null_datetime_and_complete_bounds`, `test_missing_asset_cells_convert_to_empty_item_assets`, `test_stac_json_geometry_selection_retains_one_active_geometry`, `test_record_timestamp_precision_survives_items_and_json`) and the `_item` helper. Keep `test_plain_geojson_nested_assets_remain_mappings` and `test_nullable_assets_on_an_ordinary_vector_survive_geoparquet`.

`tests/geodata/stac/test_native_items.py`: change any remaining `GeoVector.from_items(x)` to `table.from_items(x)`, and delete `test_explicit_stac_serializer_retains_item_and_asset_properties` if it asserts on `gs.to_items` (rows to Items is not provided).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata -q`
Expected: PASS

If a test outside the lists above fails, do not edit it to pass. Stop and report the test name and the failure.

- [ ] **Step 7: Checkpoint**

Run: `uv run pytest tests/geodata tests/workflow tests/ml -q && uv run ruff check src tests`
Expected: PASS. Do not commit.

---

### Task 7: Delete the old path

**Files:**
- Delete: `src/geosave_engine/geodata/io/catalog.py`
- Modify: `src/geosave_engine/geodata/io/__init__.py`
- Modify: `src/geosave_engine/geodata/stac/asset.py` (delete `from_path`)
- Modify: `src/geosave_engine/geodata/stac/item.py` (delete `from_assets`)
- Modify: `src/geosave_engine/geodata/conventions.py`
- Test: `tests/geodata/stac/test_asset.py`, `tests/geodata/stac/test_item.py`

**Interfaces:**
- Produces: no `io.catalog`, no `asset.from_path`, no `item.from_assets`, no `RASTER_ID`, `STACK_ID`, `GROUP`, `GROUPS` in `conventions`.

- [ ] **Step 1: Confirm nothing in `src` still uses the old path**

Run:

```bash
grep -rn -E "io\.catalog|from \.? ?import catalog|import catalog|catalog\.(write|describe|read_raster|read_stack|raster_ids|stack_ids)|asset\.from_path|item\.from_assets|RASTER_ID|\bGROUPS?\b|conventions import .*STACK_ID" src --include=*.py
```

Expected: matches only inside `io/catalog.py`, `io/__init__.py`, `stac/asset.py` (`from_path` itself), `stac/item.py` (`from_assets` itself) and `conventions.py`. Any other match: stop and report it.

- [ ] **Step 2: Delete**

```bash
rm src/geosave_engine/geodata/io/catalog.py
```

`src/geosave_engine/geodata/io/__init__.py`: remove `catalog` from the import list and `__all__`. In the module docstring, delete the sentence beginning "`catalog.write` coordinates those writers" and the example line `>>> items = io.catalog.write(raster, "scene", driver="cog")`.

`src/geosave_engine/geodata/stac/asset.py`: delete `from_path`, the `rasterio` import, `from geosave_engine.geodata.io import read_raster` and `gdal_path` if unused. Update the module docstring to `"""Build native STAC Assets from rasters and where they are saved."""`.

`src/geosave_engine/geodata/stac/item.py`: delete `from_assets` and the remaining `conventions` identity imports. Remove imports only it used (`Affine`, `GeoBox`) after checking with ruff. Update the module docstring to `"""Build native STAC Items from rasters and the paths their writers returned."""`.

`src/geosave_engine/geodata/conventions.py`: delete the comment line and the four constants at the end of the file.

- [ ] **Step 3: Bring the tests in line**

`tests/geodata/stac/test_asset.py`: delete the tests of `from_path` (`test_a_cog_asset_states_what_the_file_holds`, `test_a_plain_geotiff_is_not_called_a_cog`, `test_a_folder_of_files_is_not_one_asset`, `test_an_asset_built_at_write_time_agrees_with_the_file`, and any other whose body calls `asset.from_path`). Where a deleted test asserted band or grid fields that no `from_raster` test covers, port its assertions onto `asset.from_raster(raster, path)`.

`tests/geodata/stac/test_item.py`: delete the `from_assets` tests (`test_from_assets_names_layers_and_leaves_them_unowned`, `test_from_assets_dates_timeless_assets_explicitly`, `test_an_asset_without_a_grid_cannot_place_an_item`). Port `test_items_validate_against_the_stac_schemas`, `test_a_footprint_is_built_for_a_grid_without_an_epsg_code` and `test_items_survive_the_table` onto `item.from_files` and `table.from_items`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (integration and slow tests are deselected by default; if the project's default selection includes them, report which ran).

- [ ] **Step 5: Checkpoint**

Run: `uv run ruff check . && uv run pre-commit run --all-files`
Expected: clean. Do not commit.

---

### Task 8: Documentation and a final sweep

**Files:**
- Modify: `docs/guides/architecture.md`, `src/geosave_engine/model/README.md`, `src/geosave_engine/templates/workspaces/segmentation/README.md`, `src/geosave_engine/templates/workspaces/segmentation/scripts/prepare_example.py`
- Modify: `CLAUDE.md` only if its Structure section names something that moved

- [ ] **Step 1: Find every mention of the removed API**

Run:

```bash
grep -rn -E "to_raster\(|to_stack\(|from_items|\.upsert\(|raster_ids|stack_ids|to_anchor\(|normalize\(|io\.catalog|catalog\.write|catalog\.describe|from_path\(|from_assets\(|geosave:(raster|group|groups)|datetime_column|to_items\([^)]*driver" docs/guides src/geosave_engine --include=*.md --include=*.py --include=*.yaml
```

Expected: matches only in documentation and templates.

- [ ] **Step 2: Rewrite each match to the new flow**

Use this as the reference example wherever a guide shows saving and indexing:

```python
paths = ds.gs.to_cog("data/forest")
items = ds.gs.to_items(paths)
stac.table.write(items, "data/catalog/part-0.parquet")

rows = stac.table.read("data/catalog", bbox=bounds).gs.query(anchor)
ds = stac.table.load(rows)
```

and this wherever a guide shows a training sample:

```python
sample = stack({"optical": optical, "label": label})
items = sample.gs.to_items(sample.gs.to_cog("data/train/s0"), name="s0")
stac.table.write(items, "data/train/manifest.parquet")
```

Do not edit files under `docs/superpowers/`; they are history. Ignore `.ipynb` files.

- [ ] **Step 3: Verify**

Run the grep from Step 1 again.
Expected: no match.

Run: `uv run pytest -q && uv run ruff check . && uv run pre-commit run --all-files`
Expected: PASS and clean.

- [ ] **Step 4: Report**

Report to the user, per the project's handoff format: what changed, checks run with their output, and breaking changes or remaining risks. Include:

- the integration tests dropped with `tests/geodata/core/test_catalog.py` (remote bucket and Hugging Face round trips), which have no replacement yet;
- anything stopped on under the "stop and report" rule;
- the follow-up plan to remove `StacMetadata`.

Do not commit.

---

## Self-review notes

- Spec coverage: definitions and rules 1 to 6 (Tasks 2, 3, 6); module responsibilities (Tasks 1 to 7); callers (Task 5); decisions by default 1 to 3 (Tasks 1, 3, 6); tests table (each task); breaking changes (Tasks 6, 7); documentation (Task 8). The StacMetadata follow-up is deliberately not in this plan.
- Type names used across tasks: `ItemTime` (defined in Task 2, used in Task 4), `STACK_ID` (Task 2, used in Task 5), `table.from_items/write/read/load` (Task 3, used in Tasks 4 to 6).
- Href handling moves from `io/geoparquet.py` to `stac/table.py` in one step (Task 3), so no task has two owners for it.
