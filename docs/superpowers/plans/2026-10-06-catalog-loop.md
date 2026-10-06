# Catalog Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make save, register, query and read one coherent loop: writers return
paths, `GeoVector.from_rasters(paths)` builds the catalog, and catalog rows read
back through `read_raster` / `read_stack`.

**Architecture:** Writers write pixels only. Registration opens each saved file
once and describes it as a PySTAC Asset; Items are built from Assets; the table
is serialized by stac-geoparquet. Reading hands the hrefs a catalog names to the
existing format readers, so a writer's return value is a reader's input.

**Tech Stack:** Python 3.12, xarray, GeoPandas 1.1, PySTAC 1.14, stac-geoparquet
0.8, odc-geo, rasterio/rioxarray, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-06-catalog-loop-design.md`. Read it
before starting; this plan argues from it.

## Global Constraints

- The checkout is dirty on `main` and shared with the user. Do not commit,
  stash, checkout, restore or delete files this plan does not name.
- No `sed` or regex renames. Rename with exact-string edits, one file at a time.
- No new dependencies. No compatibility aliases. No custom STAC fields.
- No private helper, adapter or module constant beyond the ones this plan
  names. If a step needs more code or branches than shown, a second code path
  for one job, or a special case to make a test pass, stop and report what grew
  and the simpler alternative before continuing.
- Dependencies point one way: `geodata <- model <- ml`. `geodata` imports no torch.
- Readers must not compute pixels while opening.
- Vocabulary: scene, asset, COG, chip, catalog (spec, "Names"). Names say what a
  value holds; no participles (`stored`, `resolved`, `planned`).
- Google-style docstrings: purpose, inputs, outputs, constraints. No design
  history. Check with `python scripts/check_docstrings.py <files>`.
- JSON through `orjson`, never stdlib `json`.
- Tests needing the network (`Item.validate()`) carry `@pytest.mark.integration`.
- Ignore notebooks.

## Review Focus

Inputs the spec implies but its examples do not show. Each has a test in the
task named.

1. A path given relative to the working directory, and a folder name holding a
   space: hrefs must be absolute in memory and still resolve after the table is
   written and read (Tasks 4 and 5).
2. A grid whose CRS has no EPSG code: the asset states `proj:wkt2` and the Item
   footprint is still built (Tasks 5 and 6).
3. A row holding non-data assets, such as a remote thumbnail: readers leave
   them alone instead of trying to open them (Task 7).
4. Sub-second instants: file names keep the fraction, and the instant read back
   equals the one written (Tasks 2 and 3).
5. Two sources holding one variable at one instant, such as a raster saved
   twice as COGs and as a store: xarray merges them silently and one wins, so
   the reader raises before combining (Tasks 2 and 7).

## File Structure

| File | Responsibility after this plan |
| --- | --- |
| `src/geosave_engine/geodata/io/cogs.py` (new, replaces `io/layout.py`) | Arrange a Dataset as COG files |
| `src/geosave_engine/geodata/io/__init__.py` | `read_raster` for one or several sources; `read_stack` for a store, folder or named sources |
| `src/geosave_engine/geodata/io/geoparquet.py` | Table I/O; relative and absolute hrefs; STAC `bbox` |
| `src/geosave_engine/geodata/io/storage.py` | Filesystem resolution only |
| `src/geosave_engine/geodata/stac/asset.py` | One saved file to one `pystac.Asset` |
| `src/geosave_engine/geodata/stac/item.py` | Assets to Items |
| `src/geosave_engine/geodata/attrs/headers/stac.py` | attrs and STAC band field names, both directions |
| `src/geosave_engine/geodata/core/vector.py` | `from_rasters`, `to_raster`, `query` |
| `src/geosave_engine/geodata/core/row.py` | `hrefs`, `to_raster`, `to_stack` |
| `src/geosave_engine/geodata/core/{raster,stack,array}.py` | Writers without `catalog=` / `id=` / `layout=` |
| `src/geosave_engine/geodata/utils/datetime.py` | The one instant token |
| `src/geosave_engine/geodata/transform/vector.py` | `chip_windows` |
| `src/geosave_engine/workflow/tasks/labels.py`, `workflow/flows/ingest.py`, `cli/commands/workflow/ingest.py` | Callers |

---

### Task 1: Remove the reserved `catalog=` and `id=` arguments

**Files:**
- Modify: `src/geosave_engine/geodata/core/raster.py` (`to_cog`, `to_zarr`, `to_netcdf`)
- Modify: `src/geosave_engine/geodata/core/stack.py` (`to_cog`, `to_zarr`, `to_netcdf`)
- Modify: `src/geosave_engine/geodata/core/array.py` (`to_cog`)
- Modify: `src/geosave_engine/workflow/flows/ingest.py`, `src/geosave_engine/cli/commands/workflow/ingest.py`
- Delete: `tests/geodata/io/test_catalog.py`
- Test: `tests/geodata/core/test_raster.py`, `tests/workflow/flows/test_ingest.py`, `tests/cli/commands/test_workflow.py`

**Interfaces:**
- Produces: seven writers and the `ingest` flow with no `catalog` parameter.
  `id` stays only on `GeoRaster.to_cog` until Task 3 removes it.

- [x] **Step 1: Write the failing test** in `tests/geodata/core/test_raster.py`

```python
import inspect

from geosave_engine.geodata import GeoArray, GeoRaster, GeoStack


@pytest.mark.parametrize(
    "writer",
    [
        GeoRaster.to_cog, GeoRaster.to_zarr, GeoRaster.to_netcdf,
        GeoStack.to_cog, GeoStack.to_zarr, GeoStack.to_netcdf,
        GeoArray.to_cog,
    ],
)
def test_writers_take_no_catalog_argument(writer) -> None:
    assert "catalog" not in inspect.signature(writer).parameters


@pytest.mark.parametrize(
    "writer",
    [
        GeoRaster.to_zarr, GeoRaster.to_netcdf,
        GeoStack.to_cog, GeoStack.to_zarr, GeoStack.to_netcdf,
        GeoArray.to_cog,
    ],
)
def test_writers_that_name_nothing_take_no_id(writer) -> None:
    assert "id" not in inspect.signature(writer).parameters
```

In `tests/workflow/flows/test_ingest.py` replace
`test_ingest_catalog_is_explicitly_pending_before_loading` with:

```python
def test_ingest_takes_no_catalog_argument() -> None:
    assert "catalog" not in inspect.signature(ingest.fn).parameters
```

In `tests/cli/commands/test_workflow.py` drop `"--catalog"` from the option
tuple at line 26, the two argument lines at 76-77 and the expected
`"catalog": ...` entry at line 88.

- [x] **Step 2: Run** `uv run pytest tests/geodata/core/test_raster.py -k "catalog or names_nothing" tests/workflow/flows/test_ingest.py tests/cli/commands/test_workflow.py -q`
  Expected: the new tests FAIL (`catalog` is a parameter).

- [x] **Step 3: Remove the arguments.** In each of the seven writers delete the
  `catalog` parameter (and `id`, except on `GeoRaster.to_cog`), its `Args:` line,
  the `NotImplementedError` line under `Raises:`, and the guard
  `if catalog is not None: raise NotImplementedError(...)`. In the flow delete
  the `catalog` parameter, its docstring lines and the guard; in the CLI delete
  the `catalog` option and the `catalog=` keyword. Delete
  `tests/geodata/io/test_catalog.py`.

- [x] **Step 4: Run** the command from Step 2, then
  `uv run pytest tests/geodata/core tests/geodata/io tests/workflow tests/cli -q`
  Expected: PASS.

---

### Task 2: Read several sources as one raster, and named sources as one stack

**Files:**
- Modify: `src/geosave_engine/geodata/io/__init__.py`
- Create: `tests/geodata/io/test_read_raster.py`
- Modify: `tests/geodata/io/test_read_stack.py`

**Interfaces:**
- Produces:
  `read_raster(source: str | PathLike[str] | Sequence[str | PathLike[str]], **options) -> Dataset`
  and `read_stack(source: str | PathLike[str] | Mapping[str, RasterSource], **options) -> DataTree`,
  with `type RasterSource = str | PathLike[str] | Sequence[str | PathLike[str]]`
  defined in `io/__init__.py`.
- `layout.read_tree` stays in place, unused, until Task 3 deletes the module.

- [x] **Step 1: Write the failing tests** in `tests/geodata/io/test_read_raster.py`

```python
"""Several saved rasters read as one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave_engine.geodata import io, read_raster
from geosave_engine.geodata.attrs import GeoTIFFTags, rebase
from geosave_engine.geodata.errors import DroppedAttrsWarning
from geosave_engine.geodata.io import geotiff, zarr

from tests.geodata.conftest import build_raster


def write_scenes(folder: Path, cube: xr.Dataset, *, split: bool = False) -> list[Path]:
    """Write one COG per scene, or per scene and band, under arbitrary names."""
    paths = []
    for position in range(cube.sizes["time"]):
        scene = cube.isel(time=position)
        parts = [scene[[name]] for name in scene.data_vars] if split else [scene]
        for part in parts:
            name = f"file{len(paths)}.tif"
            paths.append(geotiff.write_cog(part, folder / name))
    return paths


@pytest.mark.parametrize("split", [False, True])
def test_several_files_read_as_one_lazy_raster(tmp_path: Path, split: bool) -> None:
    written = build_raster(times=2)
    paths = write_scenes(tmp_path, written, split=split)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        restored = read_raster(paths, chunks={})

    assert started == []
    assert restored.red.chunks is not None
    assert restored.gs.geobox == written.gs.geobox
    np.testing.assert_array_equal(restored.time, written.time)
    for name in written.data_vars:
        np.testing.assert_array_equal(restored[name], written[name])


def test_a_folder_reads_like_its_files(tmp_path: Path) -> None:
    paths = write_scenes(tmp_path, build_raster(times=2))

    with read_raster(tmp_path) as folder, read_raster(sorted(paths)) as files:
        xr.testing.assert_identical(folder, files)


def test_one_file_in_a_sequence_keeps_a_time_dimension(tmp_path: Path) -> None:
    (path,) = write_scenes(tmp_path, build_raster(times=1))

    assert read_raster([path]).sizes["time"] == 1
    assert "time" not in read_raster(path).dims


def test_files_and_a_store_read_through_one_call(tmp_path: Path) -> None:
    written = build_raster(times=3)
    store = zarr.write(written.isel(time=[0, 1]), tmp_path / "early.zarr")
    late = geotiff.write_cog(written.isel(time=2), tmp_path / "late.tif")

    restored = read_raster([store, late])

    np.testing.assert_array_equal(restored.time, written.time)
    np.testing.assert_array_equal(restored.red, written.red)


def test_a_sub_second_instant_reads_back_exactly(tmp_path: Path) -> None:
    instant = np.datetime64("2025-06-01T10:30:31.123456789", "ns")
    written = build_raster(times=1).assign_coords(time=[instant])
    (path,) = write_scenes(tmp_path, written)

    assert read_raster([path]).time.values[0] == instant


def test_sources_on_two_grids_refuse(tmp_path: Path) -> None:
    here = build_raster(times=1)
    there = here.assign_coords(x=here.x + 1000)
    paths = [
        *write_scenes(tmp_path / "here", here),
        *write_scenes(tmp_path / "there", there),
    ]

    with pytest.raises(ValueError, match="different grids"):
        read_raster(paths)


def test_two_sources_holding_one_instant_refuse(tmp_path: Path) -> None:
    written = build_raster(times=2)
    store = zarr.write(written, tmp_path / "all.zarr")
    first = geotiff.write_cog(written.isel(time=0), tmp_path / "first.tif")

    with pytest.raises(ValueError, match="one variable at one instant"):
        read_raster([first, store])


def test_no_source_refuses() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        read_raster([])


def test_a_folder_holding_no_tiff_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holds no .tif"):
        read_raster(tmp_path)


@pytest.mark.parametrize("failure", [None, "open", "combine"])
def test_a_combined_read_closes_every_file(tmp_path, monkeypatch, failure) -> None:
    for name in ("a.tif", "b.tif"):
        (tmp_path / name).touch()
    closed = []

    def read(path, **options):
        path = Path(path)
        if failure == "open" and path.stem == "b":
            raise ValueError("unreadable file")
        raster = build_raster(times=2).isel(time=0 if path.stem == "a" else 1)
        raster.set_close(lambda: closed.append(path.stem))
        return raster

    monkeypatch.setattr(io.gdal, "read", read)
    if failure == "combine":

        def combine(*args, **kwargs):
            raise ValueError("incompatible files")

        monkeypatch.setattr(io.xr, "combine_by_coords", combine)
    paths = [tmp_path / "a.tif", tmp_path / "b.tif"]
    if failure:
        with pytest.raises(ValueError):
            read_raster(paths)
        assert sorted(closed) == (["a"] if failure == "open" else ["a", "b"])
    else:
        raster = read_raster(paths)
        assert closed == []
        raster.close()
        assert sorted(closed) == ["a", "b"]


def test_a_tag_the_files_agree_on_survives(tmp_path: Path) -> None:
    cube = rebase(build_raster(times=2), GeoTIFFTags(TIFFTAG_ARTIST="geosave"))

    assert read_raster(write_scenes(tmp_path, cube)).attrs["TIFFTAG_ARTIST"] == "geosave"


def test_a_tag_the_files_state_differently_drops_and_warns(tmp_path: Path) -> None:
    cube = build_raster(times=2)
    paths = [
        geotiff.write_cog(
            rebase(cube.isel(time=position), GeoTIFFTags(TIFFTAG_ARTIST=f"artist{position}")),
            tmp_path / f"{position}.tif",
        )
        for position in range(2)
    ]

    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_ARTIST"):
        restored = read_raster(paths)

    assert "TIFFTAG_ARTIST" not in restored.attrs
```

Append to `tests/geodata/io/test_read_stack.py`:

```python
def test_named_sources_open_one_group_each(tmp_path: Path) -> None:
    label = geotiff.write_cog(_flat()[["nir"]], tmp_path / "a.tif")
    optical = zarr.write(build_raster(times=2)[["red"]], tmp_path / "b.zarr")

    sample = read_stack({"label": label, "optical": optical})

    assert sample.gs.groups == ("label", "optical")
    assert sample.gs.rasters["optical"].sizes["time"] == 2


def test_a_group_named_by_several_files_joins_them(tmp_path: Path) -> None:
    cube = build_raster(times=2)[["red"]]
    paths = [
        geotiff.write_cog(cube.isel(time=position), tmp_path / f"{position}.tif")
        for position in range(2)
    ]

    sample = read_stack({"optical": paths})

    assert sample.gs.rasters["optical"].sizes["time"] == 2
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/io/test_read_raster.py tests/geodata/io/test_read_stack.py -q`
  Expected: the new tests FAIL (a list is not a path; a dict is not a path).

- [x] **Step 3: Implement** in `src/geosave_engine/geodata/io/__init__.py`.

Add imports and the alias:

```python
from collections.abc import Mapping, Sequence

import xarray as xr

from geosave_engine.geodata.attrs import merge, rebase
from geosave_engine.geodata.core.profile import TIME_COORDINATE

from .storage import StorageOptions, absolute_location

type RasterSource = str | PathLike[str] | Sequence[str | PathLike[str]]
```

Change `read_raster` to take `source: RasterSource`, document the sequence form
in its docstring (several sources read as one raster on one grid, `time` always
a dimension, variables in source order; `ValueError` for no source, two grids,
or one variable held twice at one instant), and open the body with:

```python
    if not isinstance(source, (str, PathLike)):
        sources = list(source)
        if not sources:
            raise ValueError("read_raster needs at least one source")
        with ExitStack() as opened:
            rasters: list[xr.Dataset] = []
            for path in sources:
                raster = read_raster(path, **options)
                opened.callback(raster.close)
                # A file dates itself with a scalar; several files join along the axis.
                if TIME_COORDINATE in raster.coords and TIME_COORDINATE not in raster.dims:
                    raster = raster.expand_dims(TIME_COORDINATE)
                rasters.append(raster)
            grid = rasters[0].gs.geobox
            if any(raster.gs.geobox != grid for raster in rasters[1:]):
                raise ValueError(
                    "the sources sit on different grids; read each raster on its "
                    "own, or reproject them onto one grid first"
                )
            # xarray lets one source win a repeated plane without reading pixels.
            planes = [
                (name, instant)
                for raster in rasters
                for name in raster.data_vars
                for instant in (
                    raster[TIME_COORDINATE].values.tolist()
                    if TIME_COORDINATE in raster.coords
                    else [None]
                )
            ]
            if len(set(planes)) != len(planes):
                raise ValueError(
                    "two sources hold one variable at one instant; read the "
                    "files of one raster, each instant once"
                )
            # Attrs drop here and are rebased below, where each model rules on its own.
            cube = xr.combine_by_coords(
                rasters, compat="no_conflicts", join="exact", combine_attrs="drop"
            )
            cube = rebase(cube, merge(rasters))
            cube.set_close(opened.pop_all().close)
            return cast("Dataset", cube)
```

Replace the directory branch (`return layout.read_tree(source, **options)`) with:

```python
        paths = sorted(Path(str(source)).rglob("*.tif"))
        if not paths:
            raise ValueError(f"{source} holds no .tif file to read")
        cube = read_raster(paths, **options)
        cube.encoding["source"] = absolute_location(source)
        return cube
```

Change `read_stack` to take `source: str | PathLike[str] | Mapping[str, RasterSource]`,
document the mapping form, and open the body with:

```python
    if isinstance(source, Mapping):
        from geosave_engine.geodata.core.stack import stack

        with ExitStack() as opened:
            rasters = {}
            for name, paths in source.items():
                rasters[name] = read_raster(paths, **options)
                opened.callback(rasters[name].close)
            tree = stack(rasters)
            tree.set_close(opened.pop_all().close)
            return tree
```

In its directory branch keep the entry filter and the two `ValueError` checks,
and replace the `with ExitStack()` block that follows them with:

```python
        return read_stack({entry.stem: entry for entry in entries}, **options)
```

`join="exact"` replaces `read_tree`'s `"outer"` on purpose: a band missing at one
instant must raise, not fill with NaN. If an existing test needs `"outer"`, stop
and report it.

- [x] **Step 4: Run** `uv run pytest tests/geodata/io tests/geodata/core/test_stack.py -q`
  Expected: PASS, including the untouched `test_layout.py`.

---

### Task 3: One COG writer, and paths that name their raster

**Files:**
- Create: `src/geosave_engine/geodata/io/cogs.py`
- Delete: `src/geosave_engine/geodata/io/layout.py`, `tests/geodata/io/test_layout.py`
- Modify: `src/geosave_engine/geodata/utils/datetime.py` (`format_instant`)
- Modify: `src/geosave_engine/geodata/core/raster.py` (`to_cog`), `core/stack.py` (`to_cog`)
- Modify: `src/geosave_engine/geodata/io/__init__.py`, `src/geosave_engine/geodata/__init__.py`, `src/geosave_engine/__init__.py` (exports), `src/geosave_engine/geodata/io/safe.py:50` (message)
- Create: `tests/geodata/io/test_cogs.py`
- Modify: `tests/geodata/core/test_raster.py`, `tests/geodata/core/test_stack.py`
- Test: the file that already tests `format_stem_dates` (find it with `grep -rln format_stem_dates tests`); create `tests/geodata/utils/test_datetime.py` if none exists

**Interfaces:**
- Consumes: `read_raster(paths)`, `read_stack(mapping)` from Task 2.
- Produces:
  - `format_instant(value: pd.Timestamp) -> str`
  - `io.cogs.write(ds, path, *, split_bands=False, map_scale=None, overwrite=False, **options) -> tuple[Path, ...]`
  - `GeoRaster.to_cog(path, *, split_bands=False, map_scale=None, overwrite=False, **options) -> tuple[Path, ...]`
  - `GeoStack.to_cog(destination, *, split_bands=False, map_scale=None, overwrite=False, **options) -> dict[str, tuple[Path, ...]]`

- [x] **Step 1: Write the failing tests.**

`format_instant`:

```python
def test_format_instant_keeps_a_sub_second_fraction() -> None:
    assert format_instant(pd.Timestamp("2025-06-01T10:30:31")) == "20250601T103031"
    assert (
        format_instant(pd.Timestamp("2025-06-01T10:30:31.123456789"))
        == "20250601T103031_123456789"
    )
```

`tests/geodata/io/test_cogs.py`:

```python
"""A raster arranged as COG files named after its path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.attrs import CFVariable, rebase
from geosave_engine.geodata.io import cogs

from tests.geodata.conftest import build_raster

DAYS = ("20250601T000000", "20250602T000000")


@pytest.mark.parametrize(
    ("times", "split_bands", "expected"),
    [
        (0, False, ["scene.v2.tif"]),
        (0, True, ["scene.v2/red.tif", "scene.v2/nir.tif"]),
        (2, False, [f"scene.v2/scene.v2_{day}.tif" for day in DAYS]),
        (
            2,
            True,
            [f"scene.v2/scene.v2_{day}/{band}.tif" for day in DAYS for band in ("red", "nir")],
        ),
    ],
    ids=["timeless-joined", "timeless-split", "dated-joined", "dated-split"],
)
def test_write_names_every_file_after_its_path(
    tmp_path: Path, times: int, split_bands: bool, expected: list[str]
) -> None:
    written = build_raster(times=times)
    unrelated = tmp_path / "scene.v2" / "unrelated.tif"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"not produced by this export")

    paths = cogs.write(written, tmp_path / "scene.v2", split_bands=split_bands)

    assert paths == tuple(tmp_path / name for name in expected)
    assert all(path.is_file() for path in paths)
    restored = read_raster(paths)
    assert restored.gs.geobox == written.gs.geobox
    for name, variable in written.data_vars.items():
        np.testing.assert_array_equal(restored[name].squeeze(), variable)


def test_one_variable_is_never_split(tmp_path: Path) -> None:
    written = build_raster(times=2)[["red"]]

    paths = cogs.write(written, tmp_path / "scene", split_bands=True)

    assert paths == tuple(tmp_path / f"scene/scene_{day}.tif" for day in DAYS)


def test_a_scalar_instant_is_named_by_the_path(tmp_path: Path) -> None:
    written = build_raster(times=1).isel(time=0)

    (path,) = cogs.write(written, tmp_path / "scene")

    assert path == tmp_path / "scene.tif"
    np.testing.assert_array_equal(read_raster(path).time, written.time)


def test_a_name_ending_in_tif_stays_a_name(tmp_path: Path) -> None:
    assert cogs.write(build_raster(), tmp_path / "image.tif") == (
        tmp_path / "image.tif.tif",
    )


def test_split_band_names_with_dots_remain_distinct(tmp_path: Path) -> None:
    written = build_raster().rename_vars({"red": "band.v1", "nir": "band.v2"})

    paths = cogs.write(written, tmp_path / "scene", split_bands=True)

    assert paths == (tmp_path / "scene/band.v1.tif", tmp_path / "scene/band.v2.tif")


def test_sub_second_instants_keep_their_fraction_in_the_name(tmp_path: Path) -> None:
    instant = np.datetime64("2025-06-01T10:30:31.123456789", "ns")
    written = build_raster(times=1).assign_coords(time=[instant])

    (path,) = cogs.write(written, tmp_path / "scene")

    assert path.name == "scene_20250601T103031_123456789.tif"
    assert read_raster([path]).time.values[0] == instant


def test_joined_bands_keep_the_variable_order(tmp_path: Path) -> None:
    written = build_raster(times=2)

    paths = cogs.write(written, tmp_path / "scene")

    assert list(read_raster(paths).data_vars) == list(written.data_vars)


def test_map_scale_reaches_every_file(tmp_path: Path) -> None:
    written = rebase(
        build_raster(times=2), CFVariable(units="reflectance"), target=["red", "nir"]
    )

    paths = cogs.write(written, tmp_path / "scene", map_scale=10_000)

    for path in paths:
        with rasterio.open(path) as src:
            assert float(src.tags()["TIFFTAG_XRESOLUTION"]) == pytest.approx(10.0)
    assert read_raster(paths).red.attrs["units"] == "reflectance"


def test_an_existing_file_refuses_without_overwrite(tmp_path: Path) -> None:
    written = build_raster(times=1)
    cogs.write(written, tmp_path / "scene")

    with pytest.raises(FileExistsError):
        cogs.write(written, tmp_path / "scene")
    assert cogs.write(written, tmp_path / "scene", overwrite=True)


def test_a_raster_spanning_more_than_time_refuses(tmp_path: Path) -> None:
    spanning = build_raster(times=2).expand_dims({"model": ["a", "b"]})

    with pytest.raises(ValueError, match="a COG holds one instant of one grid"):
        cogs.write(spanning, tmp_path / "scene")
```

In `tests/geodata/core/test_raster.py` replace the three `to_cog` tests:

```python
def test_cog_export_returns_readable_files(
    tmp_path: Path, times: int, split_bands: bool, count: int
) -> None:
    written = build_raster(times=times)

    paths = written.gs.to_cog(tmp_path / "scene.v2", split_bands=split_bands)

    assert isinstance(paths, tuple)
    assert len(paths) == count
    for path in paths:
        with gdal.read(path) as restored:
            expected = written.sel(time=restored.time) if times else written
            assert restored.gs.geobox == written.gs.geobox
            assert len(restored.data_vars) == (1 if split_bands else 2)
            for name in restored.data_vars:
                np.testing.assert_array_equal(restored[name], expected[name])


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".TIF"])
def test_cog_export_to_a_tiff_path_writes_that_file(tmp_path: Path, suffix: str) -> None:
    written = build_raster(times=1).isel(time=0)
    destination = tmp_path / f"scene.v2{suffix}"

    assert written.gs.to_cog(destination) == (destination,)
    with gdal.read(destination) as restored:
        np.testing.assert_array_equal(restored.red, written.red)


def test_a_tiff_path_cannot_split_bands(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="joined bands"):
        build_raster().gs.to_cog(tmp_path / "scene.tif", split_bands=True)


def test_to_cog_takes_no_id() -> None:
    assert "id" not in inspect.signature(GeoRaster.to_cog).parameters
```

Keep the `(times, split_bands, count)` parametrization above the first test.
In `tests/geodata/core/test_stack.py` change the expected lists to tuples in
`test_timeless_cog_groups_with_file_suffixes_do_not_collide` and add:

```python
def test_a_stack_reads_back_from_what_its_cog_export_returns(tmp_path: Path) -> None:
    image = build_raster(times=2)
    written = build_stack({"image": image, "label": (image // 1000).astype("uint8")})

    restored = read_stack(written.gs.to_cog(tmp_path / "scene"))

    assert restored.gs.groups == ("image", "label")
    np.testing.assert_array_equal(
        restored.gs.rasters["label"].red, written.gs.rasters["label"].red
    )
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/io/test_cogs.py tests/geodata/core/test_raster.py tests/geodata/core/test_stack.py -q`
  Expected: FAIL (`io.cogs` does not exist).

- [x] **Step 3: Implement.**

`format_instant` in `utils/datetime.py` (add `import pandas as pd` if absent):

```python
def format_instant(value: pd.Timestamp) -> str:
    """Compact filename token for one instant, as ESA stamps its granules.

    Args:
        value: Instant at up to nanosecond precision.

    Returns:
        Token shaped `YYYYMMDDTHHMMSS`, followed by `_` and nine fraction
        digits where the instant carries sub-second precision.

    Examples:
        >>> format_instant(pd.Timestamp("2025-06-01T10:30:31"))
        '20250601T103031'
    """
    token = value.strftime("%Y%m%dT%H%M%S")
    fraction = value.microsecond * 1000 + value.nanosecond
    return f"{token}_{fraction:09d}" if fraction else token
```

`src/geosave_engine/geodata/io/cogs.py`:

```python
"""How a raster is arranged as Cloud Optimized GeoTIFF files.

A COG holds one `(band, y, x)` array: one instant of one grid, its bands the
data variables. A raster with a time dimension is therefore one scene per file
or folder, each named after the path the raster was written to.

Examples:
    `write(ds, "samples/forest")` for a raster over two instants::

        split_bands=False                      split_bands=True
        forest/                                forest/
          forest_20250601T103031.tif             forest_20250601T103031/
          forest_20250611T103031.tif               B04.tif
                                                   B08.tif
                                                 forest_20250611T103031/
                                                   B04.tif
                                                   B08.tif

    Without a time dimension the same calls give `forest.tif`, or
    `forest/B04.tif` and `forest/B08.tif`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Unpack, cast

import pandas as pd
import xarray as xr

from geosave_engine.geodata.core.profile import TIME_COORDINATE
from geosave_engine.geodata.utils.datetime import format_instant

from .geotiff import COGWriteOptions, write_cog

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset


def write(
    ds: xr.Dataset,
    path: str | PathLike[str],
    *,
    split_bands: bool = False,
    map_scale: float | None = None,
    overwrite: bool = False,
    **options: Unpack[COGWriteOptions],
) -> tuple[Path, ...]:
    """Write a raster as COG files named after `path`.

    Args:
        ds: Raster to write.
        path: Name of the raster on disk. It is never read as a filename:
            `.tif` is appended to it or to the files inside it.
        split_bands: True gives each variable its own single-band file. A
            raster with one variable is written as one file either way.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace files that already exist.
        **options: COG creation options passed to every file.

    Returns:
        Exactly the files written, in time then variable order.

    Raises:
        FileExistsError: A file exists and `overwrite` is false.
        ValueError: `ds` carries no locatable grid, spans a non-spatial axis
            other than time, or its instants are empty, repeated or NaT.

    Examples:
        >>> write(ds, "samples/forest")[0]
        PosixPath('samples/forest/forest_20250601T103031.tif')
    """
    spatial = ds.odc.spatial_dims
    if spatial is None:
        raise ValueError(
            "the raster carries no locatable grid, so its files cannot be "
            "georeferenced; assign a CRS with odc.geo.xr.assign_crs first"
        )
    # Only pixels have to fit a file; an axis only a coordinate spans writes nothing.
    pixel_dims = {str(dim) for array in ds.data_vars.values() for dim in array.dims}
    extra_dims = sorted(pixel_dims - {*spatial, TIME_COORDINATE})
    if extra_dims:
        raise ValueError(
            f"a COG holds one instant of one grid, but the raster also spans "
            f"{extra_dims}; select those axes away before writing"
        )

    root = Path(path)
    if TIME_COORDINATE in ds.dims:
        times = pd.DatetimeIndex(ds[TIME_COORDINATE].values)
        if times.hasnans or not times.is_unique or len(times) == 0:
            raise ValueError("scene times must be nonempty, unique and not NaT")
        scenes = [
            (root / f"{root.name}_{format_instant(instant)}", ds.isel({TIME_COORDINATE: index}))
            for index, instant in enumerate(times)
        ]
    else:
        scenes = [(root, ds)]

    # One variable is already one band per file, and its file keeps the scene's name.
    split = split_bands and len(ds.data_vars) > 1
    cogs: list[tuple[Path, xr.Dataset]] = []
    for scene_path, scene in scenes:
        if split:
            cogs.extend((scene_path / f"{name}.tif", scene[[name]]) for name in scene.data_vars)
        else:
            cogs.append((scene_path.with_name(f"{scene_path.name}.tif"), scene))
    return tuple(
        write_cog(
            cast("Dataset", scene), target, map_scale=map_scale, overwrite=overwrite, **options
        )
        for target, scene in cogs
    )
```

`GeoRaster.to_cog` in `core/raster.py`:

```python
    def to_cog(
        self,
        path: str | PathLike[str],
        *,
        split_bands: bool = False,
        map_scale: float | None = None,
        overwrite: bool = False,
        **options: Unpack[COGWriteOptions],
    ) -> tuple[Path, ...]:
        """Write this raster as Cloud Optimized GeoTIFFs named after `path`.

        Args:
            path: Name of the raster on disk, or one `.tif` / `.tiff` file for
                a single scene with joined bands.
            split_bands: Write a separate COG for each band.
            map_scale: Map denominator for TIFF resolution tags.
            overwrite: Replace existing files.
            **options: COG creation options.

        Returns:
            Exactly the files written, in time then variable order.

        Raises:
            ValueError: A TIFF path is asked to split bands or hold several
                instants, or the raster cannot be arranged as COGs.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> ds.gs.to_cog("samples/forest")[0]
            PosixPath('samples/forest/forest_20250601T103031.tif')
        """
        from geosave_engine.geodata.io import cogs, geotiff

        if Path(path).suffix.lower() in (".tif", ".tiff"):
            if split_bands:
                raise ValueError(
                    f"{path} names one file, which holds joined bands; pass a "
                    f"path without a TIFF suffix to split them"
                )
            return (
                geotiff.write_cog(
                    self._data, path, map_scale=map_scale, overwrite=overwrite, **options
                ),
            )
        return cogs.write(
            self._data,
            path,
            split_bands=split_bands,
            map_scale=map_scale,
            overwrite=overwrite,
            **options,
        )
```

`GeoStack.to_cog` in `core/stack.py`: drop `layout`, return
`dict[str, tuple[Path, ...]]`, update the docstring (no `layout`, no `KeyError`,
example `paths["dem"]` is `(PosixPath('scene/dem.tif'),)`), and make the body:

```python
        from geosave_engine.geodata.io import cogs

        return {
            name: cogs.write(
                raster,
                Path(destination) / name,
                split_bands=split_bands,
                map_scale=map_scale,
                overwrite=overwrite,
                **options,
            )
            for name, raster in self.rasters.items()
        }
```

Remove the `LeafPath` import from `core/stack.py`. Delete `io/layout.py`. In
`io/__init__.py` import `cogs` instead of `layout`, drop
`LAYOUTS, LeafPath, read_tree, write_tree` from the imports and `__all__`, add
`"cogs"`, and update the module docstring and `read_raster`'s `Args:` (a folder
of COGs, not "a directory `write_tree` wrote"). Drop `LAYOUTS` and `write_tree`
from `geodata/__init__.py` and `geosave_engine/__init__.py`. Reword
`io/safe.py:50` to name `read_raster`. Delete `tests/geodata/io/test_layout.py`.

- [x] **Step 4: Run** `uv run pytest tests/geodata tests/workflow tests/cli -q && uv run ruff check src tests`
  Expected: PASS. A failure in a test that still expects the old file names
  (`grep -rn "20250601T000000" tests`) is updated to the new names.

---

### Task 4: Catalog table: relative hrefs and `bbox`

**Files:**
- Modify: `src/geosave_engine/geodata/io/geoparquet.py`, `io/storage.py`, `io/__init__.py` (`read_vector`)
- Modify: `tests/geodata/stac/test_native_items.py`, `tests/geodata/io/test_geoparquet.py`, `tests/geodata/io/test_reference.py`, `tests/geodata/io/test_storage.py`

**Interfaces:**
- Produces: `geoparquet.relative_hrefs(frame, table: str) -> gpd.GeoDataFrame`,
  `geoparquet.absolute_hrefs(frame, table: str) -> gpd.GeoDataFrame`, where
  `table` is `absolute_location(path)`. `storage.stored_asset_path` and
  `storage.resolve_asset_path` no longer exist.

- [x] **Step 1: Write the failing tests** in `tests/geodata/stac/test_native_items.py`

```python
def test_a_stac_table_keeps_its_bbox_through_read_and_rewrite(tmp_path):
    import pyarrow.parquet as pq
    import pytest
    from stac_geoparquet.arrow import stac_table_to_items

    from geosave_engine.geodata import read_vector

    entry, _ = _item(tmp_path)
    first = GeoVector.from_items([entry]).gs.to_geoparquet(
        tmp_path / "catalog.parquet", stac=True
    )
    second = read_vector(first).gs.to_geoparquet(tmp_path / "again.parquet", stac=True)

    for path in (first, second):
        (restored,) = stac_table_to_items(pq.read_table(path))
        assert restored["bbox"] == pytest.approx(entry.bbox)


def test_an_edited_geometry_changes_the_bbox_written(tmp_path):
    import pyarrow.parquet as pq
    from stac_geoparquet.arrow import stac_table_to_items

    entry, _ = _item(tmp_path)
    frame = GeoVector.from_items([entry])
    frame = frame.set_geometry(frame.geometry.translate(xoff=1.0))

    (restored,) = stac_table_to_items(
        pq.read_table(frame.gs.to_geoparquet(tmp_path / "catalog.parquet", stac=True))
    )

    assert restored["bbox"][0] == pytest.approx(entry.bbox[0] + 1.0)


@pytest.mark.parametrize("folder", ["dataset", "data set"])
def test_a_stac_table_follows_its_assets_when_moved(tmp_path, folder):
    import shutil

    import pyarrow.parquet as pq

    from geosave_engine.geodata import read_vector

    home = tmp_path / folder
    home.mkdir()
    entry, _ = _item(home)
    table = GeoVector.from_items([entry]).gs.to_geoparquet(
        home / "catalog.parquet", stac=True
    )
    stored = pq.read_table(table).column("assets")[0].as_py()
    assert stored["reflectance"]["href"] == "./scene.tif"
    assert stored["thumbnail"]["href"] == "https://example.com/thumb.jpg"

    moved = tmp_path / "moved"
    shutil.move(home, moved)

    href = read_vector(moved / "catalog.parquet").iloc[0].assets["reflectance"]["href"]
    assert href == str(moved / "scene.tif")
    assert Path(href).is_file()
```

(`import pytest` and `from pathlib import Path` go at the top of the file.)
In `tests/geodata/io/test_geoparquet.py` change the stored href at line 151 to
`"./rasters/near.zarr"`, and in `tests/geodata/io/test_reference.py` change
`"image.nc"` at line 31 to `"./image.nc"`. In `tests/geodata/io/test_storage.py`
delete the three tests that call `resolve_asset_path` / `stored_asset_path` and
their imports.

- [x] **Step 2: Run** `uv run pytest tests/geodata/stac/test_native_items.py tests/geodata/io/test_geoparquet.py tests/geodata/io/test_reference.py -q`
  Expected: FAIL (no `bbox` after rewrite; absolute hrefs in STAC tables).

- [x] **Step 3: Implement** in `io/geoparquet.py`.

Imports: add `from functools import partial` and
`from pystac.utils import make_absolute_href, make_relative_href`; import
`absolute_location` from `.storage` in place of `resolve_asset_path` and
`stored_asset_path`; drop `TYPE_CHECKING`'s `AbstractFileSystem`.

Replace `stored_assets` and `resolved_assets` with:

```python
def relative_hrefs(frame: gpd.GeoDataFrame, table: str) -> gpd.GeoDataFrame:
    """Rewrite asset hrefs relative to the table that stores them.

    Args:
        frame: Table carrying `assets` with absolute hrefs.
        table: Absolute path or URL of the table.

    Returns:
        Copy whose hrefs on the table's own filesystem are relative to it.

    Raises:
        TypeError: An href is neither a string nor path-like.
    """
    assets = []
    for cell in frame["assets"]:
        # A cell is one row's assets, or null where the row has none.
        if not isinstance(cell, Mapping):
            assets.append(cell)
            continue
        row = {}
        for key, asset in cell.items():
            if asset is None:
                continue
            href = asset["href"]
            if not isinstance(href, (str, PathLike)):
                raise TypeError(
                    f"asset href must be string or path-like, got {type(href).__name__}"
                )
            row[key] = {**asset, "href": make_relative_href(str(href), table)}
        assets.append(row)
    return cast("gpd.GeoDataFrame", frame.assign(assets=assets))


def absolute_hrefs(frame: gpd.GeoDataFrame, table: str) -> gpd.GeoDataFrame:
    """Rewrite the hrefs a table stores into directly openable ones.

    Args:
        frame: Table as read, carrying `assets`.
        table: Absolute path or URL the table was read from.

    Returns:
        Table whose rows hold only the assets they have, each with an
        absolute href.
    """
    assets = []
    for cell in frame["assets"]:
        if not isinstance(cell, Mapping):
            assets.append(cell)
            continue
        row = {}
        for key, asset in cell.items():
            # Parquet stores one struct for every row, null where a row has none.
            if asset is None:
                continue
            fields = {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in asset.items()
                if value is not None
            }
            row[key] = {**fields, "href": make_absolute_href(fields["href"], table)}
        assets.append(row)
    return cast("gpd.GeoDataFrame", frame.assign(assets=assets))
```

Delete `_write_stac`. In `write`, replace everything from `frame = gdf` to the
end with:

```python
    frame = relative_hrefs(gdf, absolute_location(path)) if "assets" in gdf else gdf
    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_options = cast("dict[str, Any]", dict(write_options.pop("parquet_options", {})))
    if stac:
        # GeoPandas drops the covering bbox on read, so every write states the current one.
        bounds = frame.geometry.bounds.set_axis(["xmin", "ymin", "xmax", "ymax"], axis=1)
        table = frame.assign(bbox=bounds.to_dict("records")).to_arrow(
            index=write_options.pop("index", None)
        )
        save = partial(to_parquet, table, **write_options, **parquet_options)
    else:
        save = partial(frame.to_parquet, **write_options, **parquet_options)

    if is_local_filesystem(filesystem):
        local_target = Path(target)
        if local_target.exists() and not overwrite:
            raise FileExistsError(f"{local_target} exists; pass overwrite=True to replace it")
        temp = local_target.with_name(f".{local_target.stem}.staging{local_target.suffix}")
        try:
            save(temp)
            os.replace(temp, local_target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return Path(target) if protocol is not None else Path(path)

    exists = filesystem.exists(target)
    if exists and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    try:
        save(target, filesystem=filesystem)
    except BaseException:
        if not exists:
            # Cleanup is best-effort and must not replace the write error.
            with suppress(Exception):
                filesystem.rm(target)
        raise
    return str(path)
```

Update `write`'s docstring: asset hrefs are stored relative to the table for
both kinds of table; `stac=True` states `bbox` from the geometry written.

In `io/__init__.py` `read_vector`, replace the `if "assets" in frame:` block with:

```python
        if "assets" in frame:
            frame = geoparquet.absolute_hrefs(frame, absolute_location(source))
```

and drop the now unused `filesystem_path` import there. In `io/storage.py`
delete `resolve_asset_path`, `stored_asset_path` and the imports only they used.

- [x] **Step 4: Run** `uv run pytest tests/geodata/io tests/geodata/stac tests/workflow -q`
  Expected: PASS. A remote-table test that expected `memory:///...` with three
  slashes now gets pystac's spelling; update the expectation to what
  `make_absolute_href` returns and confirm the href opens.

---

### Task 5: Describe a saved file as an Asset

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/headers/stac.py` (`band_fields`)
- Modify: `src/geosave_engine/geodata/stac/asset.py` (replace `from_xarray` with `from_path`)
- Modify: `tests/geodata/stac/test_asset.py` (replace contents)

**Interfaces:**
- Produces:
  - `attrs.headers.stac.band_fields(variable: xr.DataArray) -> dict[str, object]`
    with keys `name`, `data_type`, and, where present, `nodata`, `unit`,
    `description`, `scale`, `offset`.
  - `stac.asset.from_path(path: str | PathLike[str]) -> pystac.Asset`. The
    asset's `common_metadata.start_datetime` / `end_datetime` are UTC datetimes,
    equal for a file with a scalar instant, absent for a timeless file.

- [x] **Step 1: Write the failing tests** (`tests/geodata/stac/test_asset.py`)

```python
"""An Asset states what one saved raster file holds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import io, raster, read_raster
from geosave_engine.geodata.stac.asset import from_path

from tests.geodata.conftest import build_raster


def test_a_cog_asset_states_what_the_file_holds(tmp_path: Path) -> None:
    source = build_raster(times=1, packed=True).isel(time=0)
    source["red"].attrs.update(units="1", long_name="Red reflectance")
    path = io.geotiff.write_cog(source, tmp_path / "scene.tif")

    fields = from_path(path).to_dict()

    assert fields["href"] == str(path)
    assert fields["type"] == pystac.MediaType.COG
    assert fields["roles"] == ["data"]
    assert fields["proj:code"] == "EPSG:32749"
    assert fields["proj:shape"] == [2, 2]
    assert fields["proj:transform"] == list(source.gs.geobox.transform)[:6]
    assert [band["name"] for band in fields["eo:bands"]] == ["red", "nir"]
    assert fields["eo:bands"][0]["description"] == "Red reflectance"
    red = fields["raster:bands"][0]
    assert (red["data_type"], red["nodata"], red["unit"]) == ("uint16", 0, "1")
    assert red["scale"] == pytest.approx(1e-4)
    assert fields["start_datetime"] == fields["end_datetime"] == "2025-06-01T00:00:00Z"
    orjson.dumps(fields)


def test_the_asset_describes_stored_values_not_decoded_ones(tmp_path: Path) -> None:
    first = io.geotiff.write_cog(
        build_raster(times=1, packed=True).isel(time=0), tmp_path / "packed.tif"
    )
    with read_raster(first, mask_and_scale=True) as decoded:
        assert decoded.red.dtype.kind == "f"
        second = io.geotiff.write_cog(decoded, tmp_path / "again.tif")

    red = from_path(second).to_dict()["raster:bands"][0]

    assert red["data_type"] == "uint16"
    assert red["scale"] == pytest.approx(1e-4)


def test_a_plain_geotiff_is_not_called_a_cog(tmp_path: Path) -> None:
    path = io.geotiff.write_gtiff(build_raster(), tmp_path / "plain.tif")

    assert from_path(path).media_type == pystac.MediaType.GEOTIFF


@pytest.mark.parametrize(
    ("write", "name", "media_type"),
    [
        (io.zarr.write, "cube.zarr", pystac.MediaType.ZARR),
        (io.netcdf.write, "cube.nc", pystac.MediaType.NETCDF),
    ],
)
def test_a_store_asset_spans_its_time_axis(tmp_path, write, name, media_type) -> None:
    source = build_raster(times=2)
    path = write(source, tmp_path / name)

    asset = from_path(path)

    start, end = source.gs.timespan
    assert asset.media_type == media_type
    assert asset.common_metadata.start_datetime.replace(tzinfo=None) == start
    assert asset.common_metadata.end_datetime.replace(tzinfo=None) == end
    assert [band["name"] for band in asset.to_dict()["eo:bands"]] == ["red", "nir"]


def test_a_timeless_file_states_no_time(tmp_path: Path) -> None:
    asset = from_path(io.geotiff.write_cog(build_raster(), tmp_path / "dem.tif"))

    assert asset.common_metadata.start_datetime is None


def test_a_relative_path_becomes_an_absolute_href(tmp_path, monkeypatch) -> None:
    io.geotiff.write_cog(build_raster(), tmp_path / "scene.tif")
    monkeypatch.chdir(tmp_path)

    assert from_path("scene.tif").href == str(tmp_path / "scene.tif")


def test_a_grid_without_an_epsg_code_states_its_wkt(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=5 +lon_0=20 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    path = io.geotiff.write_cog(
        raster({"class": np.ones((4, 4), dtype="uint8")}, grid), tmp_path / "laea.tif"
    )

    fields = from_path(path).to_dict()

    assert "proj:code" not in fields
    assert "Lambert" in fields["proj:wkt2"]


def test_a_raster_without_a_grid_refuses(tmp_path: Path) -> None:
    pixels = xr.Dataset({"image": (("y", "x"), np.arange(4).reshape(2, 2))})
    path = pixels.gs.to_netcdf(tmp_path / "pixels.nc")

    with pytest.raises(ValueError, match="no locatable grid"):
        from_path(path)
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/stac/test_asset.py -q`
  Expected: FAIL (`from_path` does not exist).

- [x] **Step 3: Implement.**

Append to `attrs/headers/stac.py` (add `import numpy as np` beside the other
imports, and move `xarray` out of `TYPE_CHECKING` only if it is needed at
runtime; it is not):

```python
def band_fields(variable: xr.DataArray) -> dict[str, object]:
    """Describe one saved variable in the names STAC gives a band.

    Args:
        variable: Variable as its file stores it.

    Returns:
        {
            "name": variable name,
            "data_type": stored dtype name,
            "nodata": fill value, where the variable declares one,
            "unit" | "description" | "scale" | "offset": the attr `_ATTR_KEYS`
                maps to that field, where the variable carries it,
        }
    """
    attrs = variable.attrs
    fields = {
        "name": str(variable.name),
        "data_type": variable.dtype.name,
        "nodata": attrs.get("_FillValue"),
        **{field: attrs.get(key) for field, key in _ATTR_KEYS.items()},
    }
    # JSON holds native numbers, and a file hands back numpy ones.
    return {
        field: value.item() if isinstance(value, np.generic) else value
        for field, value in fields.items()
        if value is not None
    }
```

Replace `src/geosave_engine/geodata/stac/asset.py` with:

```python
"""Build native STAC Assets describing saved raster files."""

from __future__ import annotations

from datetime import UTC
from os import PathLike
from pathlib import PurePosixPath

import pandas as pd
import pystac
import rasterio
from odc.geo.geobox import GeoBox
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension

from geosave_engine.geodata.attrs.headers.stac import band_fields
from geosave_engine.geodata.core.profile import TIME_COORDINATE
from geosave_engine.geodata.io import read_raster
from geosave_engine.geodata.io.storage import absolute_location

_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".zarr": pystac.MediaType.ZARR,
    ".nc": pystac.MediaType.NETCDF,
    ".nc4": pystac.MediaType.NETCDF,
    ".cdf": pystac.MediaType.NETCDF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
}


def from_path(path: str | PathLike[str]) -> pystac.Asset:
    """Describe one saved raster file or store from its own header.

    Args:
        path: File or store `read_raster` opens.

    Returns:
        Asset with an absolute href, media type, the `data` role, its grid as
        projection fields, one raster and EO band per variable, and its first
        and last instant where the file is dated.

    Raises:
        ValueError: The file carries no locatable grid.

    Examples:
        >>> from_path("samples/forest/forest_20250601T103031.tif").media_type
        'image/tiff; application=geotiff; profile=cloud-optimized'
    """
    href = absolute_location(path)
    suffix = PurePosixPath(href).suffix.lower()
    with read_raster(path) as raster:
        geobox = raster.gs.geobox
        if not isinstance(geobox, GeoBox) or geobox.crs is None:
            raise ValueError(
                f"{path} carries no locatable grid, which a STAC asset states; "
                f"keep it in an ordinary reference table instead"
            )
        bands = [band_fields(raster[name]) for name in raster.gs.variables]
        if TIME_COORDINATE not in raster.coords:
            span = None
        elif TIME_COORDINATE in raster.dims:
            span = raster.gs.timespan
        else:
            instant = pd.Timestamp(raster[TIME_COORDINATE].values).floor("us").to_pydatetime()
            span = (instant, instant)

    media_type = _MEDIA_TYPES[suffix]
    if media_type == pystac.MediaType.GEOTIFF:
        with rasterio.open(href) as src:
            if src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG":
                media_type = pystac.MediaType.COG

    asset = pystac.Asset(href, media_type=media_type, roles=["data"])
    projection = ProjectionExtension.ext(asset)
    if geobox.crs.epsg is None:
        projection.wkt2 = geobox.crs.to_wkt()
    else:
        projection.code = f"EPSG:{geobox.crs.epsg}"
    projection.shape = list(geobox.shape)
    projection.transform = list(geobox.transform)[:6]
    RasterExtension.ext(asset).bands = [
        RasterBand.create(
            data_type=DataType(band["data_type"]),
            nodata=band.get("nodata"),
            scale=band.get("scale"),
            offset=band.get("offset"),
            unit=band.get("unit"),
        )
        for band in bands
    ]
    EOExtension.ext(asset).bands = [
        Band.create(name=band["name"], description=band.get("description")) for band in bands
    ]
    if span is not None:
        asset.common_metadata.start_datetime = span[0].replace(tzinfo=UTC)
        asset.common_metadata.end_datetime = span[1].replace(tzinfo=UTC)
    return asset
```

If a Zarr or NetCDF variable keeps `_FillValue` or packing in `encoding` rather
than `attrs`, so the store test fails, stop and report it: the fix belongs in
how `band_fields` reads the header, not in a fallback inside `from_path`.

- [x] **Step 4: Run** `uv run pytest tests/geodata/stac/test_asset.py tests/geodata/attrs -q`
  Expected: PASS.

---

### Task 6: Items from assets, catalog rows from paths, and the label task

**Files:**
- Modify: `src/geosave_engine/geodata/stac/item.py` (replace `from_xarray`)
- Modify: `src/geosave_engine/geodata/core/vector.py` (`from_rasters`; class docstring example)
- Modify: `src/geosave_engine/workflow/tasks/labels.py` (`read_labels`)
- Modify: `tests/geodata/stac/test_item.py` (replace contents), `tests/geodata/core/test_catalog.py` (replace contents), `tests/workflow/tasks/test_labels.py`

**Interfaces:**
- Consumes: `stac.asset.from_path` (Task 5), `GeoRaster.to_cog(path)` (Task 3).
- Produces:
  - `stac.item.from_assets(assets: Mapping[str, pystac.Asset], *, id: str, datetime: DateTime | None = None) -> pystac.Item`
  - `stac.item.from_paths(paths: Iterable[str | PathLike[str]]) -> tuple[pystac.Item, ...]`
  - `GeoVector.from_rasters(paths: Iterable[str | PathLike[str]]) -> GeoDataFrame`

- [x] **Step 1: Write the failing tests.**

`tests/geodata/stac/test_item.py`:

```python
"""Items built from the assets of saved rasters."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from odc.geo.geobox import GeoBox
from stac_geoparquet.arrow import stac_table_to_items

from geosave_engine.geodata import GeoVector, io, raster
from geosave_engine.geodata.stac.asset import from_path
from geosave_engine.geodata.stac.item import from_assets, from_paths

from tests.geodata.conftest import build_raster

IDS = ["forest_20250601T000000", "forest_20250602T000000"]


@pytest.mark.parametrize(
    ("split_bands", "keys"), [(False, ["image"]), (True, ["nir", "red"])]
)
def test_one_write_becomes_one_item_per_scene(tmp_path, split_bands, keys) -> None:
    paths = build_raster(times=2).gs.to_cog(tmp_path / "forest", split_bands=split_bands)

    items = from_paths(paths)

    assert [item.id for item in items] == IDS
    assert [sorted(item.assets) for item in items] == [keys, keys]
    assert [item.datetime for item in items] == [
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 6, 2, tzinfo=UTC),
    ]
    assert all(len(item.bbox) == 4 for item in items)


@pytest.mark.integration
def test_items_validate_against_the_stac_schemas(tmp_path: Path) -> None:
    for item in from_paths(build_raster(times=2).gs.to_cog(tmp_path / "forest")):
        item.validate()


def test_a_store_is_one_item_named_by_its_stem(tmp_path: Path) -> None:
    source = build_raster(times=2)

    (item,) = from_paths([source.gs.to_zarr(tmp_path / "forest.zarr")])

    start, end = source.gs.timespan
    assert item.id == "forest"
    assert item.datetime is None
    assert item.common_metadata.start_datetime.replace(tzinfo=None) == start
    assert item.common_metadata.end_datetime.replace(tzinfo=None) == end
    assert list(item.assets) == ["image"]


@pytest.mark.parametrize(
    ("times", "name", "id"),
    [
        (1, "forest", "forest_20250601T000000"),
        (2, "scene.v2", "scene.v2_20250601T000000"),
    ],
)
def test_a_scene_is_named_by_the_path_its_files_share(tmp_path, times, name, id) -> None:
    for split_bands in (False, True):
        paths = build_raster(times=times).gs.to_cog(
            tmp_path / str(split_bands) / name, split_bands=split_bands
        )
        assert from_paths(paths)[0].id == id


def test_a_scalar_instant_file_is_named_by_its_stem(tmp_path: Path) -> None:
    paths = build_raster(times=1).isel(time=0).gs.to_cog(tmp_path / "label")

    (item,) = from_paths(paths)

    assert (item.id, item.datetime) == ("label", datetime(2025, 6, 1, tzinfo=UTC))


def test_files_of_two_rasters_in_one_call_refuse(tmp_path: Path) -> None:
    cube = build_raster(times=1)
    paths = [*cube.gs.to_cog(tmp_path / "forest"), *cube.gs.to_cog(tmp_path / "water")]

    with pytest.raises(ValueError, match="one call per raster"):
        from_paths(paths)


def test_a_timeless_file_refuses_and_names_the_way_out(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="from_assets"):
        from_paths(build_raster().gs.to_cog(tmp_path / "dem"))


def test_from_assets_names_layers_and_leaves_them_unowned(tmp_path: Path) -> None:
    optical = from_path(build_raster(times=2).gs.to_zarr(tmp_path / "optical.zarr"))
    label = from_path(build_raster(times=1).isel(time=0).gs.to_cog(tmp_path / "label")[0])

    item = from_assets({"optical": optical, "label": label}, id="s0")

    assert item.id == "s0"
    assert list(item.assets) == ["optical", "label"]
    assert item.datetime is None
    assert item.common_metadata.start_datetime == optical.common_metadata.start_datetime
    assert item.common_metadata.end_datetime == optical.common_metadata.end_datetime
    assert optical.owner is None and label.owner is None


def test_from_assets_dates_timeless_assets_explicitly(tmp_path: Path) -> None:
    dem = from_path(build_raster().gs.to_cog(tmp_path / "dem")[0])
    instant = datetime(2020, 1, 1, tzinfo=UTC)

    assert from_assets({"dem": dem}, id="dem", datetime=instant).datetime == instant
    with pytest.raises(ValueError, match="datetime="):
        from_assets({"dem": dem}, id="dem")


def test_a_footprint_is_built_for_a_grid_without_an_epsg_code(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=5 +lon_0=20 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    label = raster({"class": np.ones((4, 4), dtype="uint8")}, grid).assign_coords(
        time=np.datetime64("2025-01-15T12:00:00")
    )
    path = io.geotiff.write_cog(label, tmp_path / "laea.tif")

    item = from_assets({"label": from_path(path)}, id="laea")

    assert item.bbox[0] == pytest.approx(20.0, abs=0.01)
    assert item.bbox[1] == pytest.approx(5.0, abs=0.01)


def test_items_survive_the_table(tmp_path: Path) -> None:
    items = from_paths(build_raster(times=2).gs.to_cog(tmp_path / "forest", split_bands=True))

    table = GeoVector.from_items(items).gs.to_geoparquet(
        tmp_path / "catalog.parquet", stac=True
    )

    restored = list(stac_table_to_items(pq.read_table(table)))
    assert [entry["id"] for entry in restored] == IDS
    assert [band["name"] for band in restored[0]["assets"]["red"]["eo:bands"]] == ["red"]
```

`tests/geodata/core/test_catalog.py`:

```python
"""A catalog built from the paths a writer returns."""

from __future__ import annotations

from pathlib import Path

from geosave_engine.geodata import GeoVector

from tests.geodata.conftest import build_raster


def test_from_rasters_lists_one_row_per_scene(tmp_path: Path) -> None:
    paths = build_raster(times=2).gs.to_cog(tmp_path / "forest", split_bands=True)

    catalog = GeoVector.from_rasters(paths)

    assert catalog["id"].tolist() == ["forest_20250601T000000", "forest_20250602T000000"]
    assert catalog.crs == "EPSG:4326"
    assert sorted(catalog.iloc[0]["assets"]) == ["nir", "red"]
    assert catalog.iloc[0]["assets"]["red"]["href"] == str(paths[0])


def test_rasters_register_into_one_catalog(tmp_path: Path) -> None:
    cube = build_raster(times=1)
    forest = GeoVector.from_rasters(cube.gs.to_cog(tmp_path / "forest"))
    water = GeoVector.from_rasters([cube.gs.to_zarr(tmp_path / "water.zarr")])

    catalog = forest.gs.upsert(water, on="id")

    assert sorted(catalog["id"]) == ["forest_20250601T000000", "water"]
```

In `tests/workflow/tasks/test_labels.py` replace
`test_directory_item_construction_is_explicitly_pending` with:

```python
def test_a_label_directory_becomes_one_row_per_label(tmp_path: Path) -> None:
    first = _label(tmp_path / "north" / "a.tif")
    _label(tmp_path / "south" / "b.tif")

    labels = read_labels(tmp_path)

    assert labels["id"].tolist() == ["north/a", "south/b"]
    assert labels.iloc[0]["assets"]["label"]["href"] == str(first)
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/stac/test_item.py tests/geodata/core/test_catalog.py tests/workflow/tasks/test_labels.py -q`
  Expected: FAIL (`from_assets`, `from_paths`, `from_rasters` do not exist).

- [x] **Step 3: Implement.**

Replace `src/geosave_engine/geodata/stac/item.py` with:

```python
"""Build native STAC Items from the assets of saved rasters."""

from __future__ import annotations

import posixpath
from collections.abc import Iterable, Mapping
from datetime import datetime as DateTime
from os import PathLike
from pathlib import PurePosixPath

import pystac
from affine import Affine
from odc.geo.geobox import GeoBox
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension
from shapely.geometry import mapping

from . import asset as stac_asset


def from_assets(
    assets: Mapping[str, pystac.Asset],
    *,
    id: str,
    datetime: DateTime | None = None,
) -> pystac.Item:
    """Build one Item from named assets.

    Args:
        assets: Assets from `stac.asset.from_path`, keyed as the Item names them.
        id: Item identity.
        datetime: Item instant. None takes the span its dated assets cover.

    Returns:
        Item over the first asset's grid, dated by one instant or by a start
        and end, holding copies of the assets.

    Raises:
        ValueError: No asset is dated and `datetime` is None.

    Examples:
        >>> from_assets({"label": asset.from_path("s0/label.tif")}, id="s0").id
        's0'
    """
    projection = ProjectionExtension.ext(next(iter(assets.values())))
    geobox = GeoBox(
        tuple(projection.shape), Affine(*projection.transform[:6]), projection.crs_string
    )
    footprint = geobox.extent.to_crs("EPSG:4326").geom

    spans = [
        (times.start_datetime, times.end_datetime)
        for times in (asset.common_metadata for asset in assets.values())
        if times.start_datetime is not None
    ]
    if datetime is not None:
        start = end = datetime
    elif spans:
        start, end = min(first for first, _ in spans), max(last for _, last in spans)
    else:
        raise ValueError(
            f"none of the assets {list(assets)} is dated, and a STAC Item needs a "
            f"time; pass datetime="
        )

    item = pystac.Item(
        id,
        mapping(footprint),
        list(footprint.bounds),
        start if start == end else None,
        {},
        start_datetime=None if start == end else start,
        end_datetime=None if start == end else end,
    )
    for extension in (ProjectionExtension, RasterExtension, EOExtension):
        extension.add_to(item)
    for key, asset in assets.items():
        item.add_asset(key, asset.clone())
    return item


def from_paths(paths: Iterable[str | PathLike[str]]) -> tuple[pystac.Item, ...]:
    """Build the Items of one saved raster from the files a writer returned.

    Files sharing one instant are one scene. A file holding one variable is
    keyed by that variable, any other file or store as `image`. Each Item is
    named by the path its files share: the file's stem for one file, their
    folder for several.

    Args:
        paths: Every file of one write.

    Returns:
        One Item per scene, or one for a store spanning several instants.

    Raises:
        ValueError: A file is timeless, or two files claim one asset key or
            two scenes one id, as files of several rasters do.

    Examples:
        >>> [item.id for item in from_paths(ds.gs.to_cog("samples/forest"))]
        ['forest_20250601T103031', 'forest_20250611T103031']
    """
    scenes: dict[tuple[DateTime, DateTime], dict[str, pystac.Asset]] = {}
    for path in paths:
        asset = stac_asset.from_path(path)
        times = asset.common_metadata
        if times.start_datetime is None:
            raise ValueError(
                f"{path} carries no time, and a STAC Item needs one; build its "
                f"Item with from_assets(..., datetime=...)"
            )
        names = [band.name for band in EOExtension.ext(asset).bands]
        key = names[0] if len(names) == 1 else "image"
        scene = scenes.setdefault((times.start_datetime, times.end_datetime), {})
        if key in scene:
            raise ValueError(
                f"{path} and {scene[key].href} both hold {key!r} at one time; "
                f"register files of several rasters with one call per raster"
            )
        scene[key] = asset

    items = []
    for assets in scenes.values():
        hrefs = [asset.href for asset in assets.values()]
        shared = PurePosixPath(posixpath.commonpath(hrefs))
        items.append(from_assets(assets, id=shared.stem if len(hrefs) == 1 else shared.name))
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"the files name their scenes {ids}, which repeat; register files of "
            f"several rasters with one call per raster"
        )
    return tuple(items)
```

In `core/vector.py` add beside `from_items`:

```python
    @classmethod
    def from_rasters(cls, paths: Iterable[str | PathLike[str]]) -> GeoDataFrame:
        """Build catalog rows from the files of one saved raster.

        Args:
            paths: Every file one writer call returned.

        Returns:
            One row per scene, or one for a store spanning several instants,
            named by the path its files share. Rasters sharing a catalog need
            different names, since `upsert` replaces rows by id.

        Raises:
            ValueError: A file is timeless, or the files belong to several
                rasters.

        Examples:
            >>> catalog = GeoVector.from_rasters(ds.gs.to_cog("samples/forest"))
            >>> catalog["id"].tolist()
            ['forest_20250601T103031', 'forest_20250611T103031']
        """
        from geosave_engine.geodata.stac import item

        return cls.from_items(item.from_paths(paths))
```

and update the class docstring example to use `GeoVector.from_rasters(paths)`.

In `workflow/tasks/labels.py` replace the directory branch's loop with:

```python
        rows = []
        for sample_id, label_path in find_labels(path, pattern).items():
            label = asset.from_path(label_path)
            if label.common_metadata.start_datetime is None:
                raise ValueError(f"Label raster has no time: {label_path}")
            rows.append(item.from_assets({"label": label}, id=sample_id))
        return GeoVector.from_items(rows)
```

Drop the `read_raster` import if nothing else uses it, and remove
"Directory indexing awaits native Asset and Item construction." and the
`NotImplementedError` line from the docstring.

- [x] **Step 4: Run** `uv run pytest tests/geodata/stac tests/geodata/core/test_catalog.py tests/workflow -q`
  Expected: PASS.

---

### Task 7: Read catalog rows back, and filter by time

**Files:**
- Modify: `src/geosave_engine/geodata/core/row.py` (`hrefs`, `to_raster`, `to_stack`)
- Modify: `src/geosave_engine/geodata/core/vector.py` (`to_raster`, `query`)
- Modify: `tests/geodata/core/test_row.py`, `tests/geodata/core/test_catalog.py`, `tests/geodata/core/test_vector.py`
- Modify: `tests/ml/segmentation/supervised/test_data.py` (remove 22 `xfail` decorators and 1 `skip`)

**Interfaces:**
- Consumes: `read_raster(sequence)`, `read_stack(mapping)` (Task 2);
  `GeoVector.from_rasters`, `stac.item.from_assets` (Task 6).
- Produces: `GeoRow.hrefs -> dict[str, str]`;
  `GeoRow.to_raster(*, layer=None, **options) -> Dataset`;
  `GeoRow.to_stack(*, layers=None, **options) -> DataTree`;
  `GeoVector.to_raster(**options) -> Dataset`.

- [x] **Step 1: Write the failing tests.**

Append to `tests/geodata/core/test_catalog.py`:

```python
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave_engine.geodata import read_raster, read_vector


def _catalog(tmp_path: Path, source: xr.Dataset, **options) -> Path:
    paths = source.gs.to_cog(tmp_path / "forest", **options)
    return GeoVector.from_rasters(paths).gs.to_geoparquet(
        tmp_path / "catalog.parquet", stac=True
    )


@pytest.mark.parametrize("split_bands", [False, True])
def test_a_registered_raster_reads_back_through_a_query(tmp_path, split_bands) -> None:
    source = build_raster(times=2, packed=True)
    table = _catalog(tmp_path, source, split_bands=split_bands)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        restored = read_vector(table).gs.query(source).gs.to_raster()

    assert started == []
    assert restored.red.chunks is not None
    with read_raster(tmp_path / "forest", chunks={}) as expected:
        xr.testing.assert_identical(restored, expected)
    np.testing.assert_array_equal(restored.red, source.red)


def test_a_one_date_target_selects_one_scene(tmp_path: Path) -> None:
    source = build_raster(times=2)
    catalog = read_vector(_catalog(tmp_path, source))

    restored = catalog.gs.query(source.isel(time=[0])).gs.to_raster()

    np.testing.assert_array_equal(restored.time, source.time[:1])


def test_a_store_row_reads_back(tmp_path: Path) -> None:
    source = build_raster(times=2)
    store = source.gs.to_zarr(tmp_path / "forest.zarr")
    table = GeoVector.from_rasters([store]).gs.to_geoparquet(
        tmp_path / "catalog.parquet", stac=True
    )

    with read_raster(store, chunks={}) as expected:
        xr.testing.assert_identical(read_vector(table).gs.to_raster(), expected)


def test_an_empty_selection_refuses(tmp_path: Path) -> None:
    catalog = read_vector(_catalog(tmp_path, build_raster(times=1)))

    with pytest.raises(ValueError, match="at least one source"):
        catalog.iloc[0:0].gs.to_raster()


def test_one_raster_registered_in_two_formats_refuses_to_read_as_one(tmp_path) -> None:
    source = build_raster(times=2)
    scenes = GeoVector.from_rasters(source.gs.to_cog(tmp_path / "forest"))
    store = GeoVector.from_rasters([source.gs.to_zarr(tmp_path / "copy.zarr")])

    with pytest.raises(ValueError, match="one variable at one instant"):
        GeoVector.concat([scenes, store]).gs.to_raster()


def test_a_written_catalog_loads_in_odc_stac(tmp_path: Path) -> None:
    import odc.stac
    import stac_geoparquet

    source = build_raster(times=2)
    catalog = read_vector(_catalog(tmp_path, source, split_bands=True))

    loaded = odc.stac.load(list(stac_geoparquet.to_item_collection(catalog)), chunks={})

    assert set(loaded.data_vars) == {"red", "nir"}
    assert loaded.odc.geobox == source.odc.geobox
    np.testing.assert_array_equal(loaded.time, source.time)
    np.testing.assert_array_equal(loaded.red, source.red)
```

Replace `test_row_asset_readers_fail_explicitly` in `tests/geodata/core/test_row.py` with:

```python
from pathlib import Path

import numpy as np
from dask.callbacks import Callback

from geosave_engine.geodata import GeoVector, io
from geosave_engine.geodata.stac import asset, item


def _sample(tmp_path: Path) -> pd.Series:
    optical = build_raster(times=2).gs.to_zarr(tmp_path / "optical.zarr")
    (label,) = build_raster(times=1).isel(time=0)[["nir"]].gs.to_cog(tmp_path / "label")
    entry = item.from_assets(
        {"optical": asset.from_path(optical), "label": asset.from_path(label)}, id="s0"
    )
    return GeoVector.from_items([entry]).iloc[0]


def test_a_sample_row_opens_its_layers_as_a_lazy_stack(tmp_path: Path) -> None:
    row = _sample(tmp_path)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        sample = row.gs.to_stack(layers=["optical", "label"])

    assert started == []
    assert sample.gs.groups == ("optical", "label")
    assert sample.gs.rasters["optical"].sizes["time"] == 2
    assert sample.gs.rasters["label"].gs.variables == ("nir",)
    # Without names the groups follow the table's asset order, which Parquet sorts.
    assert set(row.gs.to_stack().gs.groups) == {"optical", "label"}


def test_a_row_opens_one_layer_as_a_raster(tmp_path: Path) -> None:
    raster = _sample(tmp_path).gs.to_raster(layer="optical")

    assert raster.gs.variables == ("red", "nir")
    assert raster.red.chunks is not None


def test_a_missing_layer_is_a_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="dem"):
        _sample(tmp_path).gs.to_stack(layers=["dem"])


def test_a_windowed_row_reads_cropped_once(tmp_path: Path) -> None:
    row = pd.concat(
        [_sample(tmp_path), pd.Series({"row_off": 0, "col_off": 1, "height": 1, "width": 1})]
    )

    sample = row.gs.to_stack(layers="label")

    assert sample.gs.rasters["label"].sizes["x"] == 1
    assert sample.gs.rasters["label"].sizes["y"] == 1


def test_assets_that_are_not_data_are_left_alone(tmp_path: Path) -> None:
    path = io.geotiff.write_cog(build_raster(times=1).isel(time=0), tmp_path / "scene.tif")
    row = pd.Series(
        {
            "assets": {
                "image": {"href": str(path), "roles": ["data"]},
                "thumbnail": {"href": "https://example.com/t.jpg", "roles": ["thumbnail"]},
                "absent": None,
            }
        }
    )

    assert row.gs.hrefs == {"image": str(path)}
    np.testing.assert_array_equal(row.gs.to_raster().red.squeeze(), build_raster().red)
```

Add to `tests/geodata/core/test_vector.py`:

```python
def test_query_filters_time_on_a_table_with_only_datetime() -> None:
    raster = build_raster(times=2)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    frame = gpd.GeoDataFrame(
        {"id": ["first", "second"], "datetime": pd.to_datetime(raster.time.values, utc=True)},
        geometry=[footprint, footprint],
        crs="EPSG:4326",
    )

    assert frame.gs.query(raster.isel(time=[0]))["id"].tolist() == ["first"]


def test_query_reads_a_span_where_a_row_has_no_datetime() -> None:
    raster = build_raster(times=2)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    frame = gpd.GeoDataFrame(
        {
            "id": ["store", "later"],
            "datetime": pd.to_datetime([None, "2025-07-01"], utc=True),
            "start_datetime": pd.to_datetime(["2025-06-01", None], utc=True),
            "end_datetime": pd.to_datetime(["2025-06-02", None], utc=True),
        },
        geometry=[footprint, footprint],
        crs="EPSG:4326",
    )

    assert frame.gs.query(raster)["id"].tolist() == ["store"]
```

(Use the module's existing imports for `gpd`, `pd` and `build_raster`; add any
that are missing.)

In `tests/ml/segmentation/supervised/test_data.py` delete each
`@pytest.mark.xfail(...)` decorator (22) and the
`@pytest.mark.skip(reason="Native STAC asset readers are pending")` decorator.
Confirm the counts with `grep -c "pytest.mark.xfail" tests/ml/segmentation/supervised/test_data.py`
before (22) and after (0).

- [x] **Step 2: Run** `uv run pytest tests/geodata/core/test_row.py tests/geodata/core/test_catalog.py tests/geodata/core/test_vector.py tests/ml/segmentation/supervised/test_data.py -q`
  Expected: FAIL with `NotImplementedError` and the one-date query returning two rows.

- [x] **Step 3: Implement.**

`core/row.py`: replace `to_stack` and `to_raster` and add `hrefs`
(`from typing import Any` joins the imports):

```python
    @property
    def hrefs(self) -> dict[str, str]:
        """Return each data asset's href by key."""
        hrefs = {}
        for key, asset in self._data["assets"].items():
            # Parquet keeps one struct for every row, null where a row has no such asset.
            if asset is not None and (asset.get("roles") is None or "data" in asset["roles"]):
                hrefs[key] = asset["href"]
        return hrefs

    def to_raster(self, *, layer: str | None = None, **options: Any) -> xr.Dataset:
        """Read this row's assets as one lazy raster.

        Args:
            layer: Asset key to read. None reads every data asset as one raster.
            **options: Forwarded to `read_raster`. `chunks` defaults to `{}`.

        Returns:
            Dataset cropped to this row's pixel window where it has one.

        Raises:
            KeyError: `layer` names no data asset of this row.
            ValueError: The assets sit on different grids.

        Examples:
            >>> catalog.iloc[0].gs.to_raster().gs.variables
            ('B04', 'B08')
        """
        from geosave_engine.geodata.io import read_raster

        hrefs = self.hrefs
        source = list(hrefs.values()) if layer is None else hrefs[layer]
        return self.crop(read_raster(source, **{"chunks": {}, **options}))

    def to_stack(
        self, *, layers: Collection[str] | str | None = None, **options: Any
    ) -> DataTree:
        """Read this row's assets as a lazy stack, one group per asset.

        Args:
            layers: Asset keys to read. None reads every data asset.
            **options: Forwarded to `read_raster`. `chunks` defaults to `{}`.

        Returns:
            DataTree cropped to this row's pixel window where it has one.

        Raises:
            KeyError: A layer names no data asset of this row.

        Examples:
            >>> samples.iloc[0].gs.to_stack(layers=["optical", "label"]).gs.groups
            ('optical', 'label')
        """
        from geosave_engine.geodata.io import read_stack

        hrefs = self.hrefs
        names = [layers] if isinstance(layers, str) else list(hrefs if layers is None else layers)
        return self.crop(
            read_stack({name: hrefs[name] for name in names}, **{"chunks": {}, **options})
        )
```

`core/vector.py`: replace `to_raster` (`from typing import Any` joins the imports):

```python
    def to_raster(self, **options: Any) -> xr.Dataset:
        """Read the data assets of every row as one lazy raster.

        Select rows first, with pandas or `query`. Variables of split-band
        scenes come back in asset-key order; select by name where order matters.

        Args:
            **options: Forwarded to `read_raster`. `chunks` defaults to `{}`.

        Returns:
            Dataset joining the rows' scenes along `time`.

        Raises:
            ValueError: The frame holds no row, or its assets sit on different
                grids or repeat an instant.

        Examples:
            >>> catalog.gs.query(scene).gs.to_raster().sizes["time"]
            2
        """
        from geosave_engine.geodata.io import read_raster

        hrefs = [href for _, row in self._data.iterrows() for href in row.gs.hrefs.values()]
        return read_raster(hrefs, **{"chunks": {}, **options})
```

In `query`, replace the block from `span = anchor.timespan` through the time
filter with:

```python
        span = anchor.timespan
        if span is not None and "datetime" in matched:
            start, end = (pd.Timestamp(naive_utc(edge), tz="UTC") for edge in span)
            instant = pd.to_datetime(matched["datetime"], utc=True)
            first, last = (
                pd.to_datetime(matched.get(name, instant), utc=True).fillna(instant)
                for name in ("start_datetime", "end_datetime")
            )
            matched = matched.loc[(first <= end) & (last >= start)]
```

and reword its docstring: time is compared where the anchor has a timespan and
the table has `datetime`; a row is read by its span where it states one, else by
its `datetime`.

- [x] **Step 4: Run** `uv run pytest tests/geodata tests/ml tests/model tests/workflow -q`
  Expected: PASS, with no `xfail` left in `test_data.py`. If a formerly `xfail`
  test now fails for a reason other than the reader, report it instead of
  restoring the marker.

---

### Task 8: `chip_windows`, remaining names, docs and full verification

**Files:**
- Modify: `src/geosave_engine/geodata/transform/vector.py`, `src/geosave_engine/ml/segmentation/supervised/data.py`
- Modify: `tests/geodata/core/test_vector.py`, `tests/geodata/io/test_geoparquet.py`, `tests/ml/test_inputs.py`, `tests/model/encoder/test_context.py`
- Modify: `src/geosave_engine/geodata/core/vector.py` (`from_items`, `upsert` locals)
- Modify: `docs/guides/architecture.md`, `docs/guides/workflows.md`, `src/geosave_engine/templates/workspaces/segmentation/README.md`, `src/geosave_engine/model/README.md`

**Interfaces:**
- Produces: `transform.vector.chip_windows(parents, tilers, *, padding=None) -> gpd.GeoDataFrame`
  with the body, columns and errors of today's `from_tilers`.

- [x] **Step 1: Rename the function.** In each of the eight Python files above
  that mention it, replace the exact string `from_tilers` with `chip_windows`
  (one exact-string replace per file). In `transform/vector.py` also change the
  docstring's first line to "List the chip windows of each parent as rows." and
  the example to `>>> chips = chip_windows(parents, tilers)`. The columns
  (`tile_id`, `parent_id`, `raster_metadata`) and `Dataset.reference` stay.

- [x] **Step 2: Rename locals** in `core/vector.py`: in `from_items`,
  `source` becomes `item`, `entry` becomes `clone` and `resolved` becomes
  `clones`; in `upsert`, `retained` becomes `others` and `incoming` becomes
  `keys`. Behaviour is unchanged; no other function is touched.

- [x] **Step 3: Run** `uv run pytest tests/geodata tests/ml tests/model -q`
  Expected: PASS.

- [x] **Step 4: Update the docs.** In the four doc files replace every mention
  of `write_tree`, `read_tree`, `LAYOUTS`, `layout=`, `catalog=`, `--catalog`,
  `to_cog(..., id=...)` and `from_tilers` with the API in the spec, and add the
  loop from the spec's first code block to `docs/guides/architecture.md` where
  it describes persistence. Find the mentions with
  `grep -rn "write_tree\|read_tree\|LAYOUTS\|layout=\|catalog=\|--catalog\|from_tilers" docs/guides src/geosave_engine/templates src/geosave_engine/model/README.md`;
  the command must print nothing afterwards.

- [x] **Step 5: Verify the whole change.**

```bash
uv run pytest
uv run pytest -m integration tests/geodata/stac/test_item.py
uv run ruff check .
python scripts/check_docstrings.py src/geosave_engine/geodata/io/cogs.py \
  src/geosave_engine/geodata/io/__init__.py src/geosave_engine/geodata/io/geoparquet.py \
  src/geosave_engine/geodata/stac/asset.py src/geosave_engine/geodata/stac/item.py \
  src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/core/row.py \
  src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/stack.py
grep -rn "write_tree\|read_tree\|LAYOUTS\|LeafPath\|stored_assets\|resolved_assets\|from_xarray\|from_tilers" src tests
```

Expected: tests and ruff pass, the docstring check reports nothing, and the
final `grep` prints nothing. Report the test counts and any test whose
expectation changed, with the reason.

---

## Execution notes, 2026-10-06

Executed inline in the shared dirty checkout on `main`; nothing was committed.

Verification:

- `uv run pytest`: 1451 passed, 14 deselected (slow and integration).
- Slow and integration tests touching this change: the segmentation workspace
  training on the example samples, worker-process reads, and
  `Item.validate()` against the STAC schemas: 3 passed.
- `uv run ruff check src tests`: clean. `scripts/check_docstrings.py` on the
  touched modules: clean.
- No `write_tree`, `read_tree`, `LAYOUTS`, `LeafPath`, `stored_assets`,
  `resolved_assets`, `from_tilers` or STAC `from_xarray` left in `src`, `tests`
  or `docs/guides`.

Rulings made during execution:

1. Work stayed in the dirty checkout with no commits, as the Global Constraints
   require; there are no per-task commit ranges.
2. `read_raster`'s repeated-plane check reads instants from `raster.indexes`
   rather than `.values.tolist()`: numpy returns ints for nanosecond times and
   datetimes for other units, so a COG and a Zarr store never compared equal.
3. `ruff format` was applied to touched files where its diff was confined to
   this plan's lines. `tests/geodata/io/test_storage.py` was left alone because
   its one hunk is older code.
4. A remote href reads back in the spelling it was written with
   (`memory://x`), not fsspec's `unstrip_protocol` spelling (`memory:///x`).
   Both open the same object; one test expectation changed.
5. Relative hrefs in STAC tables changed two tests in
   `tests/geodata/core/test_vector.py` that Task 4's test command did not
   cover. `test_stac_tools_read_the_table` used to assert that a tool reading
   the Parquet directly sees an absolute href; it now asserts `./a.zarr` and
   that the href resolves against the table path. Cost: a third-party tool that
   does not resolve hrefs against the table location cannot open the assets.
6. The no-EPSG test CRS is `+proj=laea +lat_0=12.34 +lon_0=56.78`; the plan's
   `lat_0=5 lon_0=20` resolves to EPSG:10592 and never reached `proj:wkt2`.
7. `tests/cli/core/test_workspace.py::test_segmentation_workspace_trains_on_prepared_samples`
   (slow) was migrated to `asset.from_path` / `item.from_assets` and its
   `xfail` removed; the plan's file list had missed it.
8. The final review was a self-review. It found one defect, fixed test-first:
   a folder passed to `stac.asset.from_path` raised `KeyError('')` and now
   raises a `ValueError` naming the mistake.

Deferred minors:

- `read_stack(directory)` names a layer by `entry.stem`, which truncates a
  dotted folder name such as `image.v2`. This is older behaviour;
  `read_stack(mapping)` from a writer's return value keeps exact names.
- `read_raster` holds both the suffix dispatch and the several-source combine,
  about 75 lines. Splitting it needs a second function this plan did not name.
- `transform/vector.py` has two older two-line comment runs that
  `check_docstrings.py` flags.
