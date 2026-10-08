# `y`/`x` Spatial Dimensions Everywhere Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every raster GeoSave holds spans `("y", "x")`, so `grid_dims` and every build-then-rename disappear, and `colorize` moves to `transform` like every other pixel operation.

**Architecture:** Foreign spatial names are renamed once, at the four boundaries where they enter (NetCDF read, Zarr read, `odc.stac.load`, `odc.reproject`). The three `gs` raster accessors refuse anything else at bind time. Everything between reads `SPATIAL_DIMENSIONS`.

**Tech Stack:** xarray, odc-geo (`spatial_dims`, `xr_coords`), odc-stac, pytest.

**Spec:** none; the design was settled in conversation on 2026-10-07 and is restated under Design below.

## Design

```python
# geodata/conventions.py: the whole mechanism
def to_yx[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T: ...   # boundaries call this
def require_yx(data: xr.DataArray | xr.Dataset | xr.DataTree) -> None: ... # the accessor base calls this

# a boundary
warped = to_yx(band.odc.reproject(geobox, resampling=kernel)).rename(band.name)

# everywhere else
colored = array(..., dims=(*self.axes, BAND_DIMENSION, *SPATIAL_DIMENSIONS))
return colored                                    # no rename
```

CF identifies latitude and longitude by `units`, `standard_name` and `axis`, never by dimension name, so `y`/`x` on a geographic grid stays CF-valid. `create_header` in `attrs/headers/geobox.py` already writes those attrs.

Smoke-tested on 2026-10-07 (odc-geo as pinned in `uv.lock`):

| Claim | Result |
| --- | --- |
| `odc.reproject` onto a geographic geobox | dims `("latitude", "longitude")`, lazy |
| `odc.stac.load` | builds coords with `xr_coords(gbox)`, so same names |
| `rename` to `y`/`x` | geobox equal, dask graph kept, `grid_mapping` encoding kept |
| NetCDF and Zarr with `lat`/`lon` or `latitude`/`longitude` | read, renamed, written back: geobox equal |
| `DataTree.map_over_datasets(rename)` with root-inherited coords | works; `DataTree` itself has no `rename` |
| `odc.geo.xr.spatial_dims(data)` (not relaxed) | recognises exactly `y,x`, `latitude,longitude`, `lat,lon`; dims-only, no geobox built |
| odc's relaxed guess | unsafe: reads `(north, east, depth)` as `(east, depth)` and a CRS-less `(depth, level)` as a grid |
| `data.rio.y_dim` / `x_dim` | finds any pair labelled with CF `axis` or `standard_name`, whatever its names and position; raises `MissingSpatialDimensionError` otherwise |
| Name lookup then CF lookup, per `.gs` bind | 2 µs on `y`/`x`, 45 µs on a gridless object |
| `.odc.spatial_dims` on a fresh object | ~460 µs (builds odc state), so the accessor check uses the plain function |

## Global Constraints

- Spatial dimension names are `SPATIAL_DIMENSIONS = ("y", "x")` from `geodata/conventions.py`; never the literals in `src/`.
- No compatibility alias for `grid_dims`; it is removed (CLAUDE.md: no compatibility layers).
- No new public API. A user holding an odc-loaded object is told to call xarray's own `.rename`.
- `geodata` imports no torch; `conventions.py` may import xarray and odc-geo only.
- Renames must stay lazy; each boundary test asserts the dask graph survives.
- The working tree carries unrelated uncommitted changes. Do not commit, stash, checkout or restore; leave committing to the user.
- Ignore `.ipynb` files and `docs/superpowers/` history.
- Baseline before starting: `uv run pytest tests/geodata tests/ml tests/model -q` gives 1506 passed.

## Review Focus

1. **Unlabelled exotic spatial names** (`("north", "east")` carrying no CF `axis` or `standard_name`): a pair is recognised by a conventional name or by its CF attrs, never guessed, so this one passes the accessor check and later raises `KeyError: 'y'`. Accepted: odc's guess, the only alternative, also renames non-spatial float dims. A CF-labelled exotic pair is recognised and tested in Task 1.
2. **File handle after rename**: `Dataset.rename` returns a new object without the opener's close hook. Task 2 transfers it with `set_close`, as `zarr.read` already does; its tests call `.close()` on the result. A tree always comes back as a new object from `map_over_datasets`, so its hook is always transferred.
3. **Unaligned stack in the datamodule**: `GeoStack.grid_dims` raised a named `ValueError`; `parent.sizes["y"]` raises `KeyError`. Task 4 keeps the named error by reading the shape off `parent.gs.anchor.geobox`.
4. **A group already on `y`/`x` beside one on `latitude`/`longitude`** in one foreign tree: `to_yx` must rename per node. Covered in Task 1.
5. **`y`/`x` named but ordered `(x, y)`**: `spatial_dims` reports `("y", "x")`, so nothing is renamed and nothing is refused; order stays the constructor's job (`raster` already checks the trailing pair). No test added.

---

### Task 1: `to_yx` and `require_yx`

**Files:**
- Modify: `src/geosave_engine/geodata/conventions.py`
- Create: `tests/geodata/test_conventions.py`

**Interfaces:**
- Produces: `to_yx(data: T) -> T` for `DataArray | Dataset | DataTree`; a band or raster needing no rename comes back as the same object, a tree always as a new one. `require_yx(data: DataArray | Dataset | DataTree) -> None`, raising `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the spatial dimension convention."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.conventions import require_yx, to_yx

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def loaded(dims: tuple[str, str], name: str = "red") -> xr.Dataset:
    """Build a lazy raster the way a foreign loader names its grid."""
    built = xr.Dataset(
        {name: (dims, da.ones(tuple(GRID.shape), chunks=(1, 3)))},
        coords=xr_coords(GRID, dims=dims),
    )
    built[name].encoding["grid_mapping"] = "spatial_ref"
    return built


@pytest.mark.parametrize("dims", [("latitude", "longitude"), ("lat", "lon")])
def test_to_yx_renames_a_foreign_grid_without_computing(dims) -> None:
    result = to_yx(loaded(dims))

    assert result.red.dims == ("y", "x")
    assert result.odc.geobox == GRID
    assert isinstance(result.red.data, da.Array)
    assert result.red.encoding["grid_mapping"] == "spatial_ref"


def test_to_yx_finds_a_pair_only_its_cf_attrs_name() -> None:
    labelled = loaded(("north", "east")).expand_dims(depth=[0.5, 1.5], axis=-1)
    labelled.north.attrs["axis"] = "Y"
    labelled.east.attrs["standard_name"] = "longitude"

    result = to_yx(labelled)

    assert result.red.dims == ("y", "x", "depth")
    assert result.odc.geobox == GRID


def test_to_yx_leaves_an_unlabelled_pair_alone() -> None:
    unlabelled = loaded(("north", "east"))

    assert to_yx(unlabelled) is unlabelled


def test_to_yx_renames_one_band() -> None:
    result = to_yx(loaded(("latitude", "longitude")).red)

    assert result.dims == ("y", "x")
    assert result.odc.geobox == GRID


def test_to_yx_returns_the_same_object_when_nothing_is_foreign() -> None:
    native = loaded(("y", "x"))
    gridless = xr.Dataset({"count": (("time",), np.arange(2))})

    assert to_yx(native) is native
    assert to_yx(gridless) is gridless


def test_to_yx_renames_each_group_of_a_tree() -> None:
    foreign = loaded(("latitude", "longitude"))
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords=xr_coords(GRID)),
            "/optical": foreign,
            "/dem": loaded(("y", "x"), name="elevation"),
        }
    )

    result = to_yx(tree)

    assert result["optical"].dataset.red.dims == ("y", "x")
    assert result["dem"].dataset.elevation.dims == ("y", "x")
    assert result.dataset.odc.geobox == GRID
    assert isinstance(result["optical"].dataset.red.data, da.Array)


def test_require_yx_names_the_rename_that_fixes_a_foreign_grid() -> None:
    foreign = loaded(("latitude", "longitude"))

    with pytest.raises(ValueError, match=r"rename\(\{'latitude': 'y'"):
        require_yx(foreign)
    with pytest.raises(ValueError, match=r"rename\(\{'latitude': 'y'"):
        require_yx(xr.DataTree.from_dict({"/optical": foreign}))


def test_require_yx_accepts_native_and_gridless_objects() -> None:
    require_yx(loaded(("y", "x")))
    require_yx(xr.DataArray(np.ones(2), dims=("time",)))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/geodata/test_conventions.py -q`
Expected: collection error, `cannot import name 'require_yx'`.

- [ ] **Step 3: Implement**

In `src/geosave_engine/geodata/conventions.py`, replace the docstring's closing paragraph

```text
Factories use `y` and `x` for every CRS. Geographic coordinates carry CF
standard names `latitude` and `longitude` and degree units; projected
coordinates carry projection standard names and linear units. Loaded objects
retain their original spatial dimension names.
```

with

```text
Every raster spans `y` and `x`, whatever its CRS. Geographic coordinates carry
CF standard names `latitude` and `longitude` and degree units; projected
coordinates carry projection standard names and linear units. CF reads those
attrs, not the dimension names. Readers rename a foreign grid on the way in
with `to_yx`, and the `gs` accessors refuse one with `require_yx`.
```

Add after the docstring (the module has no imports today):

```python
from __future__ import annotations

from typing import cast

import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr
from odc.geo.xr import spatial_dims
from rioxarray.exceptions import MissingSpatialDimensionError
```

and after the constants:

```python
def _spatial_pair(data: xr.DataArray | xr.Dataset) -> tuple[str, str] | None:
    """Find the spatial pair by a conventional name, then by its CF attrs."""
    dims = spatial_dims(data)  # y, x | latitude, longitude | lat, lon
    if dims is None:
        try:
            # Any name, where the coordinates carry CF `axis` or `standard_name`.
            dims = (str(data.rio.y_dim), str(data.rio.x_dim))
        except MissingSpatialDimensionError:
            return None
    return dims


def to_yx[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Rename a foreign spatial pair to `y` and `x`.

    The pair is found by name (`latitude`/`longitude`, `lat`/`lon`) or, under
    any other name, by the CF `axis` or `standard_name` its coordinates carry.

    Args:
        data: Band, raster, or stack as a loader named its grid.

    Returns:
        Renamed view sharing its pixels, lazily. A band or raster already on
        `y` and `x`, or carrying no grid, comes back as itself.

    Examples:
        >>> to_yx(xr.open_dataset("era5.nc")).t2m.dims
        ('time', 'y', 'x')
    """
    if isinstance(data, xr.DataTree):
        return cast("T", data.map_over_datasets(to_yx))

    dims = _spatial_pair(data)
    if dims in (None, SPATIAL_DIMENSIONS):
        return data
    return cast("T", data.rename(dict(zip(dims, SPATIAL_DIMENSIONS, strict=True))))


def require_yx(data: xr.DataArray | xr.Dataset | xr.DataTree) -> None:
    """Refuse a grid spanning anything but `y` and `x`.

    Args:
        data: Band, raster, or stack about to be read through `gs`.

    Raises:
        ValueError: Its spatial dimensions, or a group's, carry another
            standard pair.
    """
    if isinstance(data, xr.DataTree):
        for node in data.subtree:
            require_yx(node.dataset)
        return

    dims = _spatial_pair(data)
    if dims not in (None, SPATIAL_DIMENSIONS):
        fix = dict(zip(dims, SPATIAL_DIMENSIONS, strict=True))
        raise ValueError(
            f"the grid spans {dims}, but gs reads {SPATIAL_DIMENSIONS} for every "
            f"CRS; rename it first with .rename({fix})"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/geodata/test_conventions.py -q`
Expected: 9 passed.

---

### Task 2: Rename at the four boundaries

**Files:**
- Modify: `src/geosave_engine/geodata/io/raster/netcdf.py` (`read`, `read_stack`)
- Modify: `src/geosave_engine/geodata/io/raster/zarr.py` (`read`, `read_stack`)
- Modify: `src/geosave_engine/geodata/stac/source.py:378`
- Modify: `src/geosave_engine/geodata/transform/warp.py:308`
- Test: `tests/geodata/io/raster/test_netcdf.py`, `tests/geodata/io/raster/test_zarr.py`, `tests/geodata/stac/test_source.py`, `tests/geodata/transform/test_warp.py`

**Interfaces:**
- Consumes: `to_yx` from Task 1.
- Produces: every reader, `StacSource.load` and `reproject` return `y`/`x` objects.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/io/raster/test_netcdf.py`:

```python
def _foreign() -> xr.Dataset:
    """Build a geographic raster named the way odc and most tools write it."""
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    built = xr.Dataset(
        {"red": (("latitude", "longitude"), np.ones(grid.shape, "uint16"))},
        coords=xr_coords(grid),
    )
    built.red.encoding["grid_mapping"] = "spatial_ref"
    return built


def test_netcdf_reads_a_foreign_grid_onto_y_x(tmp_path: Path) -> None:
    foreign = _foreign()
    foreign.to_netcdf(tmp_path / "foreign.nc")

    restored = netcdf.read(tmp_path / "foreign.nc", chunks={})

    assert restored.red.dims == ("y", "x")
    assert restored.gs.geobox == foreign.odc.geobox
    assert restored.red.chunks is not None
    restored.close()


def test_netcdf_reads_a_foreign_stack_onto_y_x(tmp_path: Path) -> None:
    foreign = _foreign()
    xr.DataTree.from_dict({"/optical": foreign}).to_netcdf(tmp_path / "foreign.nc")

    restored = netcdf.read_stack(tmp_path / "foreign.nc")

    assert restored["optical"].dataset.red.dims == ("y", "x")
    assert restored["optical"].dataset.gs.geobox == foreign.odc.geobox
    restored.close()
```

Append to `tests/geodata/io/raster/test_zarr.py`:

```python
def test_zarr_reads_a_foreign_grid_onto_y_x(tmp_path: Path) -> None:
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    foreign = xr.Dataset(
        {"red": (("latitude", "longitude"), np.ones(grid.shape, "uint16"))},
        coords=xr_coords(grid),
    )
    foreign.red.encoding["grid_mapping"] = "spatial_ref"
    foreign.to_zarr(tmp_path / "foreign.zarr")
    xr.DataTree.from_dict({"/optical": foreign}).to_zarr(tmp_path / "tree.zarr")

    restored = zarr.read(tmp_path / "foreign.zarr", chunks={})
    tree = zarr.read_stack(tmp_path / "tree.zarr")

    assert restored.red.dims == ("y", "x")
    assert restored.gs.geobox == grid
    assert restored.red.chunks is not None
    assert tree["optical"].dataset.red.dims == ("y", "x")
    restored.close()
    tree.close()
```

Append to `tests/geodata/stac/test_source.py` (next to `test_load_returns_dask_backed_arrays_by_default`, reusing its imports and `FakeClient`):

```python
def test_load_names_a_geographic_grid_y_x(monkeypatch: pytest.MonkeyPatch) -> None:
    from odc.geo.xr import xr_coords

    def fake_load(items: list[object], *, geobox: GeoBox, **options: Any) -> xr.Dataset:
        # odc-stac names a geographic grid latitude/longitude.
        coords = dict(xr_coords(geobox))
        coords["time"] = [dt(2025, 6, 1)]
        loaded = xr.Dataset(
            {"red": (("time", *geobox.dimensions), da.ones((1, *geobox.shape)))},
            coords=coords,
        )
        loaded.red.encoding["grid_mapping"] = "spatial_ref"
        return loaded

    monkeypatch.setattr(source_module.odc.stac, "load", fake_load)
    monkeypatch.setattr(
        source_module, "create_header", lambda *args, **kwargs: AttrsHeader()
    )
    anchor = GeoAnchor.from_coordinates(
        -6.5914, 107.8416, shape=2, resolution=0.0001, crs="EPSG:4326",
        timespan="2025-06",
    )
    assert anchor.geobox.dimensions == ("latitude", "longitude")

    loaded = StacSource(FakeClient(), collection="example").load(anchor)  # type: ignore[arg-type]

    assert loaded.red.dims == ("time", "y", "x")
    assert loaded.gs.geobox == anchor.geobox
    assert isinstance(loaded.red.data, da.Array)
    assert loaded.y.attrs["standard_name"] == "latitude"
```

The assertion on `anchor.geobox.dimensions` guards the premise that odc names this grid `latitude`/`longitude`.

Append to `tests/geodata/transform/test_warp.py`:

```python
def test_reproject_onto_a_geographic_grid_keeps_y_x_and_laziness() -> None:
    source = scene().chunk({"y": 2, "x": 2})

    warped = reproject(source, WGS84)

    assert warped.red.dims == ("y", "x")
    assert warped.gs.crs.epsg == 4326
    assert warped.red.chunks is not None
    assert warped.y.attrs["standard_name"] == "latitude"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/geodata/io/raster/test_netcdf.py tests/geodata/io/raster/test_zarr.py tests/geodata/stac/test_source.py tests/geodata/transform/test_warp.py -q -k "foreign or geographic"`
Expected: 5 failed, each on a dims assertion showing `('latitude', 'longitude')`.

- [ ] **Step 3: Implement**

`netcdf.py`: add `from geosave_engine.geodata.conventions import to_yx`. In `read`, replace `cube = xr.open_dataset(` … `)` and the two lines after it with:

```python
    opened = xr.open_dataset(
        source,
        engine=engine,
        # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
        decode_coords="all",
        group=group,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **open_options,
    )
    cube = to_yx(opened)
    if cube is not opened:
        cube.set_close(opened.close)
    cube.encoding["source"] = absolute_location(source)
    return cast("Dataset", cube)
```

In `read_stack`, rename the opened tree and normalise it before the encoding loop:

```python
    opened = xr.open_datatree(
        source,
        engine=engine,
        # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
        decode_coords="all",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **open_options,
    )
    stack = to_yx(opened)
    stack.set_close(opened.close)
```

`zarr.py`: add the same import. In `read` change `cube = _in_written_order(opened)` to `cube = to_yx(_in_written_order(opened))`. In `read_stack` change `stack = opened.map_over_datasets(_in_written_order)` to `stack = to_yx(opened.map_over_datasets(_in_written_order))`. The existing `if … is not opened: set_close` lines already cover the new object.

`stac/source.py`: add `from geosave_engine.geodata.conventions import to_yx` and wrap the load:

```python
        data = to_yx(
            odc.stac.load(matched, geobox=anchor.geobox, **self.config.to_load_kwargs())
        )
```

`warp.py`: add the same import and change line 308 to:

```python
    warped = to_yx(band.odc.reproject(geobox, resampling=kernel)).rename(band.name)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/geodata/io tests/geodata/stac tests/geodata/transform -q`
Expected: all pass, including the 5 new tests.

---

### Task 3: The accessors refuse a foreign grid

**Files:**
- Modify: `src/geosave_engine/geodata/core/base.py` (new `GeoRasterAccessor.__init__`); `core/array.py`, `core/raster.py`, `core/stack.py` (delete their `__init__`)
- Modify: `tests/geodata/conftest.py:36-37`, `tests/geodata/transform/conftest.py:63-68`
- Test: `tests/geodata/core/test_base.py`, `tests/geodata/core/test_raster.py`, `tests/geodata/core/test_stack.py`, `tests/geodata/transform/test_vector.py`

**Interfaces:**
- Consumes: `require_yx` from Task 1; `y`/`x` boundaries from Task 2 (without them a geographic `reproject` would trip the check from inside `_warp_band`).
- Produces: `.gs` on a DataArray, Dataset or DataTree raises `ValueError` when a grid spans `latitude`/`longitude` or `lat`/`lon`.

- [ ] **Step 1: Move the fixtures onto `y`/`x`**

`tests/geodata/conftest.py`: replace

```python
    coords = dict(xr_coords(geobox))
    spatial_dims = geobox.dimensions  # ('latitude', 'longitude') when geographic
```

with

```python
    coords = dict(xr_coords(geobox, always_yx=True))
    spatial_dims = ("y", "x")
```

`tests/geodata/transform/conftest.py`: replace both `("time", *box.dimensions)` with `("time", "y", "x")` and `coords = dict(xr_coords(box))` with `coords = dict(xr_coords(box, always_yx=True))`.

`tests/geodata/core/test_raster.py:156,173`: replace `*box.dimensions` with `"y", "x"`. `tests/geodata/core/test_stack.py:97`: replace `(*raster.gs.geobox.dimensions, "spatial_ref")` with `("y", "x", "spatial_ref")`.

- [ ] **Step 2: Replace the tests that pinned tolerance with tests that pin refusal**

Delete `test_write_crs_preserves_loaded_geographic_dimension_names` (`tests/geodata/core/test_raster.py`).

In the three tests parametrized over `[("y", "x"), ("latitude", "longitude")]`, drop the parametrize decorator and the `dims` argument, add `dims = ("y", "x")` as the first body line, and rename them:

| File | Old name | New name |
| --- | --- | --- |
| `tests/geodata/core/test_stack.py:324` | `test_shared_geographic_stack_uses_its_rasters_dimension_names` | `test_shared_geographic_stack_spans_y_x` |
| `tests/geodata/core/test_raster.py:834` | `test_colorize_preserves_geographic_spatial_names` | `test_colorize_keeps_a_geographic_grid` |
| `tests/geodata/transform/test_vector.py:294` | `test_geographic_polygonization_and_rasterization_preserve_spatial_names` | `test_geographic_polygonization_and_rasterization_round_trip` |

Append to `tests/geodata/core/test_base.py`:

```python
def _foreign() -> xr.Dataset:
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    built = xr.Dataset(
        {"red": (("latitude", "longitude"), np.ones(grid.shape))},
        coords=xr_coords(grid),
    )
    built.red.encoding["grid_mapping"] = "spatial_ref"
    return built


def test_gs_refuses_a_grid_not_named_y_x() -> None:
    foreign = _foreign()
    tree = xr.DataTree.from_dict({"/optical": foreign})

    for held in (foreign, foreign.red, tree):
        with pytest.raises(ValueError, match="rename it first"):
            _ = held.gs


def test_gs_reads_the_grid_once_it_is_renamed() -> None:
    foreign = _foreign()

    renamed = foreign.rename({"latitude": "y", "longitude": "x"})

    assert renamed.gs.geobox == foreign.odc.geobox
    assert renamed.gs.write_crs().y.attrs["standard_name"] == "latitude"

```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/geodata/core/test_base.py -q -k "refuses or renamed"`
Expected: `test_gs_refuses_a_grid_not_named_y_x` fails with `DID NOT RAISE`; the other passes.

- [ ] **Step 4: Implement**

The three accessors each carry an `__init__` that only binds `_data`. One on the base replaces them and owns the check.

`core/base.py`: import `require_yx` alongside `TIME_COORDINATE`, and add as the first member of `GeoRasterAccessor`, after `_data: DataT`:

```python
    def __init__(self, data: DataT) -> None:
        """Bind the xarray object.

        Args:
            data: Band, raster, or stack to read through this accessor.

        Raises:
            ValueError: Its grid, or a group's, spans dimensions other than
                `y` and `x`.
        """
        require_yx(data)
        self._data = data
```

Delete `GeoArray.__init__` (`core/array.py:95-101`), `GeoRaster.__init__` (`core/raster.py:207-213`) and `GeoStack.__init__` (`core/stack.py:174-180`). Remove `cast` from an import only where nothing else in the file uses it.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/geodata tests/ml tests/model -q`
Expected: all pass. Any remaining failure is a test still building `latitude`/`longitude` by hand; move it onto `("y", "x")` with `xr_coords(grid, always_yx=True)`.

---

### Task 4: Remove `grid_dims` and the renames

**Files:**
- Modify: `src/geosave_engine/geodata/core/{array,raster,stack}.py`, `attrs/headers/geobox.py`, `transform/{nodata,vector,chip,warp}.py`, `viz/plot.py`, `ml/segmentation/supervised/{data,module}.py`
- Modify tests: `tests/geodata/core/{test_raster,test_stack}.py`, `tests/geodata/transform/{test_packing,test_nodata}.py`, `tests/geodata/io/raster/{test_netcdf,test_zarr}.py`, `tests/ml/test_inputs.py`, `tests/model/encoder/test_context.py`
- Modify docs: `docs/guides/architecture.md:304`, docstring at `core/raster.py:39-45`

**Interfaces:**
- Consumes: the `y`/`x` guarantee from Tasks 2-3.
- Produces: no `grid_dims` anywhere; `create_header(geobox)` in `attrs/headers/geobox.py` takes no `dims`.

A removal has no behaviour of its own to test first: the existing suite is the test, and Step 4's grep is the proof nothing still names `grid_dims`.

- [ ] **Step 1: Delete the three properties**

Remove `grid_dims` from `GeoArray` (`core/array.py:108-117`), `GeoRaster` (`core/raster.py:215-224`) and `GeoStack` (`core/stack.py:200-221`).

- [ ] **Step 2: Replace every caller in `src/`**

Each file imports `SPATIAL_DIMENSIONS` from `geosave_engine.geodata.conventions` (add it to the existing import where one exists).

| Location | Before | After |
| --- | --- | --- |
| `core/array.py` `axes` | `grid_dims = self.grid_dims` … `if dim not in grid_dims` | `if dim not in SPATIAL_DIMENSIONS` |
| `core/array.py` `to_numpy` | `transpose(*ahead, *self.grid_dims)` | `transpose(*ahead, *SPATIAL_DIMENSIONS)` |
| `core/array.py` `colorize` | `colored = array(…)`, `y_dim, x_dim = self.grid_dims`, `return cast("DataArray", colored.rename({"y": y_dim, "x": x_dim}))` | `return array(…)` |
| `core/raster.py` `write_crs` | `create_geobox_header(geobox, dims=result.gs.grid_dims)` | `create_geobox_header(geobox)` |
| `core/raster.py` `raster` | `create_geobox_header(geobox, dims=SPATIAL_DIMENSIONS)` | `create_geobox_header(geobox)` |
| `core/raster.py` `to_array` | `grid_dims = self.grid_dims` | `grid_dims = SPATIAL_DIMENSIONS` |
| `core/stack.py` `stack` | `raster.gs.geobox != shared or raster.gs.grid_dims != reference.gs.grid_dims` | `raster.gs.geobox != shared` |
| `core/stack.py` `stack` | `for name in (*reference.gs.grid_dims, CRS_COORDINATE)` | `for name in (*SPATIAL_DIMENSIONS, CRS_COORDINATE)` |
| `transform/nodata.py:150` | `grid_dims = spatial.gs.grid_dims` | `grid_dims = SPATIAL_DIMENSIONS` |
| `transform/warp.py:455` | `grid_dims = set(raster.gs.grid_dims)` | `grid_dims = set(SPATIAL_DIMENSIONS)` |
| `transform/chip.py:63` | `dims = data.gs.grid_dims` | `dims = SPATIAL_DIMENSIONS` |
| `transform/vector.py:55-58` | `flags.gs.grid_dims` (twice) | `SPATIAL_DIMENSIONS` |
| `viz/plot.py:175,229,313` | `y, x = array.gs.grid_dims` | `y, x = SPATIAL_DIMENSIONS` |
| `ml/…/module.py:312-313` | `image.gs.grid_dims`, `labels.gs.grid_dims` | `SPATIAL_DIMENSIONS` |

`transform/vector.py:235-243` becomes:

```python
    burned = array(pixels, geobox, dims=SPATIAL_DIMENSIONS).rename(output_name)
    if isinstance(like, GeoAnchor):
        return burned
    # A sliced raster's coordinates drift from its geobox's own by float error,
    # so the result takes the target's, which is what aligns with it.
    source = like.dataset if isinstance(like, xr.DataTree) else like
    return burned.assign_coords({dim: source.coords[dim] for dim in SPATIAL_DIMENSIONS})
```

`ml/segmentation/supervised/data.py:122-123` becomes (the anchor keeps the named error for a stack with no shared grid):

```python
            tiler = self.spec.chips.tiler(tuple(parent.gs.anchor.geobox.shape))
```

`attrs/headers/geobox.py`: drop the `dims` parameter and its `Args` line, change "under the selected dimension names" to "under `y` and `x`", import `SPATIAL_DIMENSIONS`, and replace the two lines computing names and coords with:

```python
    y_dim, x_dim = SPATIAL_DIMENSIONS
    coords = xr_coords(geobox, always_yx=True)
```

- [ ] **Step 3: Replace every caller in `tests/` and docs**

| Location | Change |
| --- | --- |
| `tests/geodata/core/test_raster.py:188` | `assert built.red.dims == ("y", "x")` |
| `tests/geodata/core/test_raster.py:791` | `assert built.red.dims == ("y", "x")` |
| `tests/geodata/core/test_raster.py:807` | `assert restored.red.dims == ("y", "x")` |
| `tests/geodata/core/test_stack.py:159-160` | delete the `grid_dims` `pytest.raises` block; keep the `anchor` one; rename the test `test_a_stack_without_a_shared_grid_names_no_anchor` |
| `tests/geodata/core/test_stack.py:338-341` | `assert tree["optical"].dataset.red.dims == dims`, `assert tree["dem"].dataset.elevation.dims == dims`; keep the geobox assertion |
| `tests/geodata/transform/test_packing.py:104` | `source["dem"] = (("y", "x"), …)` |
| `tests/geodata/transform/test_nodata.py:51,176` | `dims = ("y", "x")` |
| `tests/geodata/io/raster/test_netcdf.py:25-26` | `assert restored["y"].attrs["standard_name"] == written["y"].attrs["standard_name"]` |
| `tests/geodata/io/raster/test_zarr.py:31-34` | drop the two `grid_dims` lines; index with `"y"` and `"x"` |
| `tests/ml/test_inputs.py:306` | `spatial = ("y", "x")` |
| `tests/model/encoder/test_context.py:50` | `y, x = "y", "x"` |
| `docs/guides/architecture.md:304` | `key: spec.chips.tiler(tuple(parent.gs.anchor.geobox.shape))` |

`core/raster.py:39-45` docstring: replace the `>>> wgs84.gs.grid_dims` example with

```text
    Geographic grids span `y` and `x` too. CF metadata identifies them as
    latitude and longitude:

    >>> wgs84.red.dims
    ('y', 'x')
    >>> wgs84.y.attrs["standard_name"], wgs84.y.attrs["units"]
    ('latitude', 'degrees_north')
```

- [ ] **Step 4: Verify nothing is left**

Run: `grep -rn "grid_dims\|\.rename({\"y\"" src tests docs/guides`
Expected: only the `grid_dims = SPATIAL_DIMENSIONS` locals in `to_array`, `nodata.py` and `warp.py`.

- [ ] **Step 5: Run everything**

Run: `uv run pytest tests/geodata tests/ml tests/model -q && uv run ruff check .`
Expected: 0 failed and more than the 1506 baseline passing (new tests outnumber the one deleted and three collapsed parametrizations); ruff clean. Report the actual count.

---

### Task 5: Move `colorize` to `transform`

Every other pixel operation on the accessors is a one-line delegation to `transform` (`mask`, `to_nan`, `unpack`, `reproject`, `crop`, `vectorize`). `colorize` is the one that carries its body inline.

**Files:**
- Create: `src/geosave_engine/geodata/transform/color.py`
- Modify: `src/geosave_engine/geodata/transform/__init__.py`, `src/geosave_engine/geodata/core/array.py` (`GeoArray.colorize`)
- Create: `tests/geodata/transform/test_color.py`
- Modify: `tests/geodata/core/test_raster.py` (remove `test_colorize_keeps_a_geographic_grid`)

**Interfaces:**
- Consumes: `colorize` body as Task 4 left it (returns `array(...)` directly).
- Produces: `transform.color.colorize(band: xr.DataArray) -> xr.DataArray`; `GeoArray.colorize()` unchanged for callers.

- [ ] **Step 1: Write the failing test**

Move `test_colorize_keeps_a_geographic_grid` out of `tests/geodata/core/test_raster.py` into the new `tests/geodata/transform/test_color.py`:

```python
"""Tests for baking class colours into display channels."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs import Legend, rebase
from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.transform.color import colorize

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def landcover(legend: Legend | None) -> xr.DataArray:
    """Build a class band, one pixel of it naming no class."""
    codes = np.array([[1, 1, 1], [1, 1, 9]], dtype="uint8")
    band = array(codes, GRID, dims=("y", "x"))
    return band if legend is None else rebase(band, legend)


def test_colorize_bakes_each_class_colour_onto_the_grid() -> None:
    band = landcover(Legend(class_map={1: "water"}, color_map={1: "#0000ff"}))

    colored = colorize(band)

    assert colored.dims == ("band", "y", "x")
    assert list(colored.band.values) == ["red", "green", "blue"]
    assert colored.gs.geobox == GRID
    assert colored.y.attrs["standard_name"] == "latitude"
    np.testing.assert_array_equal(colored.sel(band="blue")[0], np.ones(3))
    assert np.isnan(colored.values[:, 1, 2]).all()
    xr.testing.assert_identical(band.gs.colorize(), colored)


def test_colorize_refuses_a_band_without_classes_or_colours() -> None:
    with pytest.raises(ValueError, match="lists no classes"):
        colorize(landcover(None))
    with pytest.raises(ValueError, match="carry no colour"):
        colorize(landcover(Legend(class_map={1: "water"})))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/geodata/transform/test_color.py -q`
Expected: collection error, `No module named 'geosave_engine.geodata.transform.color'`.

- [ ] **Step 3: Implement**

Create `src/geosave_engine/geodata/transform/color.py`. The body is `GeoArray.colorize`'s, with `self._data` read as `band` and `self.axes` as `band.gs.axes`:

```python
"""Bake the colours a band's legend names into display channels."""

from __future__ import annotations

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.conventions import BAND_DIMENSION, SPATIAL_DIMENSIONS
from geosave_engine.geodata.utils.color import parse_color


def colorize(band: xr.DataArray) -> xr.DataArray:
    """Colour each pixel by the class its code names.

    Args:
        band: Class codes carrying a `Legend`.

    Returns:
        Georeferenced array shaped `(*axes, band, y, x)` valued in `[0, 1]`,
        whose `band` coordinate is `("red", "green", "blue")`. A pixel no
        class names is absent on every channel.

    Raises:
        ValueError: The band lists no classes, or names a class carrying no
            colour.

    Examples:
        >>> colorize(ds["landcover"]).sizes["band"]
        3
    """
    from geosave_engine.geodata.core.array import array

    legend = attrs.Legend.from_attrs(band.attrs)
    class_map = None if legend is None else legend.class_map
    if legend is None or class_map is None:
        raise ValueError(
            "band lists no classes, so its values name none to colour; write "
            "a Legend, or compose channels with GeoRaster.to_array"
        )

    colour_of = legend.color_map or {}
    codes = sorted(class_map)
    missing_colour = [code for code in codes if code not in colour_of]
    if missing_colour:
        raise ValueError(
            f"classes {missing_colour} carry no colour; give Legend.color_map an "
            f"entry for every class the band lists"
        )

    palette = np.array(
        [parse_color(colour_of[code]) for code in codes], dtype="float32"
    )
    palette /= 255.0  # (class, 3)

    pixels = band.values
    names_class = np.isin(pixels, codes)
    code_index = np.where(names_class, np.searchsorted(codes, pixels), 0)
    channels = np.where(names_class[..., None], palette[code_index], np.nan)

    axes = band.gs.axes
    return array(
        np.moveaxis(channels, -1, -3),  # (*axes, band, y, x)
        band.odc.geobox,
        dims=(*axes, BAND_DIMENSION, *SPATIAL_DIMENSIONS),
        nodata=None,  # absence is NaN here, which no fill value stands for
        coords={
            **{name: labels for name, labels in axes.items() if labels is not None},
            BAND_DIMENSION: ["red", "green", "blue"],
        },
    )
```

`transform/__init__.py`: add `color` to the import tuple and to `__all__`, both in alphabetical position (after `chip`).

`core/array.py`: keep `GeoArray.colorize`'s signature and docstring; its body becomes

```python
        from geosave_engine.geodata.transform import color

        return cast("DataArray", color.colorize(self._data))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/geodata tests/ml tests/model -q && uv run ruff check .`
Expected: 0 failed; ruff clean (it flags any import `array.py` no longer uses).

- [ ] **Step 5: Smoke-test the user path end to end**

```bash
uv run python - <<'EOF'
import numpy as np, xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from geosave_engine.geodata import raster
from geosave_engine.geodata.attrs import Legend, rebase

grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
scene = raster({"cls": (("y", "x"), np.ones((2, 3), "uint8"))}, grid)
band = rebase(scene.cls, Legend(class_map={1: "water"}, color_map={1: "#0000ff"}))
print(band.gs.colorize().dims)

loaded = xr.Dataset({"cls": (("latitude", "longitude"), np.ones((2, 3)))}, coords=xr_coords(grid))
try:
    loaded.gs
except ValueError as error:
    print(error)
EOF
```

Expected: `('band', 'y', 'x')`, then the message ending `rename it first with .rename({'latitude': 'y', 'longitude': 'x'})`.
