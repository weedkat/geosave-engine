# One Raster Accessor Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `GeoAccessor` and `GeoRasterAccessor` into one base, so a stack gains `unpack`, `mask`, `to_nan`, `reproject`, and `crop`, and bring the transforms those methods call onto one typing style.

**Architecture:** One helper, `map_groups`, applies a change to every group of a stack and restores the stack's root attrs. `packing.unpack`, `nodata.to_nan`, and `vector.crop` become generic over `DataArray | Dataset | DataTree` and use it for the stack case; `nodata.mask` switches to it. With every transform accepting a stack, the two accessor bases collapse into `GeoRasterAccessor`, whose methods return `DataT` with no `cast`.

**Tech Stack:** Python 3.12, xarray, odc-geo, rasterio, GeoPandas, pytest, basedpyright, Ruff

**Spec:** `docs/superpowers/specs/2026-10-04-geovector-accessor-design.md`, Part A. Part B (the `GeoVector` accessor) gets its own plan.

## Global Constraints

- No existing behaviour changes on a `Dataset` or a `DataArray`. Every existing test passes unmodified.
- `crop` still raises when the vector's CRS differs from the raster's. Changing that rule belongs to Part B.
- `mosaic`, `merge_bands`, `reduce`, `resample`, `interpolate`, `frames`, and `tiling`'s per-tile rebuild stay as they are.
- `geodata` imports no torch at module level. `tensor()` keeps importing it inside the function.
- Transforms import `map_groups` inside the function body, as `warp.reproject` already imports `stack`. A module-level import would cycle through `core/array.py`.
- Do not commit, stage, stash, checkout, or restore anything. Every file this plan touches already carries the user's uncommitted work; leave all changes in the working tree.
- Run commands with `uv run`. Do not touch `.ipynb` files.
- Docstrings are concise Google style. Comments explain domain constraints only.

## Review Focus

- A stack whose groups sit on different grids publishes no grid at its root. `map_groups` must still rebuild it and keep its root attrs. Task 1 pins this.
- A stack operation must stay lazy: no pixel is computed by `unpack`, `to_nan`, or `mask` on a chunked stack. Task 2 pins this.
- `crop` on a raster with a `time` axis must mask every instant, since the burned mask spans only the grid. Task 3 pins this.
- `crop` with an empty vector must raise a `ValueError`, not return an empty raster. Task 3 pins this.
- `stack.gs.reproject(raster)` must adopt that raster's grid whole, unlike `align`, whose default extent is the union. Task 4 pins this.

## File Structure

| File | Change |
| --- | --- |
| `src/geosave_engine/geodata/core/stack.py` | Add `map_groups`; `GeoStack` inherits the merged base |
| `src/geosave_engine/geodata/transform/packing.py` | `unpack` generic, accepts a stack |
| `src/geosave_engine/geodata/transform/nodata.py` | `to_nan` generic, accepts a stack; `mask` uses `map_groups` |
| `src/geosave_engine/geodata/transform/vector.py` | `crop` accepts a stack, burns with `rasterize`, reads `footprint` |
| `src/geosave_engine/geodata/core/base.py` | One class, no `cast`, no `tensor` |
| `src/geosave_engine/geodata/core/array.py` | New home of `tensor` |
| `src/geosave_engine/geodata/core/raster.py` | Imports `tensor` from `array` |
| `src/geosave_engine/geodata/transform/tiling.py` | Type fix in `_map_groups` |
| `src/geosave_engine/geodata/transform/time.py` | Type fix in `_joint_axis` |
| `tests/geodata/core/test_stack.py` | `map_groups` and stack accessor tests |
| `tests/geodata/transform/test_packing.py` | Stack tests for `unpack` and `to_nan` |
| `tests/geodata/transform/test_nodata.py` | Root attrs survive a stack `mask` |
| `tests/geodata/transform/test_vector.py` | Stack, time-axis, burner, and empty-vector tests for `crop` |

---

### Task 1: `map_groups`

**Files:**
- Modify: `src/geosave_engine/geodata/core/stack.py`
- Test: `tests/geodata/core/test_stack.py`

**Interfaces:**
- Consumes: `stack(rasters: Mapping[str, xr.Dataset]) -> DataTree`, `tree.gs.rasters -> dict[str, Dataset]`, `tree.gs.attrs.root`, `attrs.rebase(data, namespace)`, all existing.
- Produces: `map_groups(tree: xr.DataTree, change: Callable[[xr.Dataset], xr.Dataset]) -> DataTree` in `geosave_engine.geodata.core.stack`.

- [ ] **Step 1: Write the failing tests**

In `tests/geodata/core/test_stack.py`, add `import geosave_engine.geodata.attrs as attrs` to the imports, change the stack import to

```python
from geosave_engine.geodata.core.stack import map_groups, stack as build_stack
```

and append:

```python
def test_map_groups_changes_every_group_and_keeps_the_root_attrs() -> None:
    raster = build_raster()
    built = build_stack({"optical": raster[["red"]], "dem": raster[["nir"]]})
    titled = built.gs.rebase(attrs.ACDD(title="scene"))

    halved = map_groups(titled, lambda group: group.isel(x=slice(0, 1)))

    assert halved.gs.groups == ("optical", "dem")
    assert halved.gs.geobox.shape.x == 1
    assert halved.gs.attrs.root.get(attrs.ACDD).title == "scene"


def test_map_groups_rebuilds_a_stack_that_shares_no_grid() -> None:
    raster = build_raster()
    coarse = raster[["nir"]].isel(x=slice(0, 1))
    built = build_stack({"optical": raster[["red"]], "dem": coarse})
    titled = built.gs.rebase(attrs.ACDD(title="scene"))

    renamed = map_groups(
        titled, lambda group: group.rename({name: f"{name}_raw" for name in group.data_vars})
    )

    assert renamed.gs.geobox is None
    assert renamed.gs.variables == ("optical/red_raw", "dem/nir_raw")
    assert renamed.gs.attrs.root.get(attrs.ACDD).title == "scene"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_stack.py -q`
Expected: collection error, `ImportError: cannot import name 'map_groups'`.

- [ ] **Step 3: Implement**

In `src/geosave_engine/geodata/core/stack.py`, add below `from geosave_engine.geodata.transform import warp`:

```python
import geosave_engine.geodata.attrs as attrs
```

Add `from collections.abc import Callable` inside the `if TYPE_CHECKING:` block.

Add directly below the `stack` function:

```python
def map_groups(
    tree: xr.DataTree, change: Callable[[xr.Dataset], xr.Dataset]
) -> DataTree:
    """Apply one change to every group, keeping the stack's own attrs.

    The stack is rebuilt rather than mapped over, so a change that moves the
    grid still leaves a root that matches its groups.

    Args:
        tree: Stack to change.
        change: Returns the changed raster for one group.

    Returns:
        New stack of the changed groups, in the same order, carrying the
        root attrs `tree` carried.

    Examples:
        >>> map_groups(scene, lambda raster: raster.isel(time=0)).gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    changed = stack({name: change(raster) for name, raster in tree.gs.rasters.items()})
    return attrs.rebase(changed, tree.gs.attrs.root)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/core/test_stack.py -q`
Expected: all pass.

---

### Task 2: `unpack`, `to_nan`, and `mask` on a stack

**Files:**
- Modify: `src/geosave_engine/geodata/transform/packing.py`
- Modify: `src/geosave_engine/geodata/transform/nodata.py:29-62` and `:145-153`
- Test: `tests/geodata/transform/test_packing.py`
- Test: `tests/geodata/transform/test_nodata.py`

**Interfaces:**
- Consumes: `map_groups(tree, change) -> DataTree` from Task 1.
- Produces:
  - `unpack[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T`
  - `to_nan[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T`
  - `mask` keeps its signature.

- [ ] **Step 1: Write the failing tests**

In `tests/geodata/transform/test_packing.py`, add to the imports:

```python
from geosave_engine.geodata.core.stack import stack as build_stack
```

and append:

```python
def test_a_stack_unpacks_group_by_group_and_stays_lazy() -> None:
    tree = build_stack(
        {"optical": stored(1000, scale=1e-4).chunk(), "dem": stored(7, scale=None).chunk()}
    ).gs.rebase(attrs.ACDD(title="scene"))

    physical = unpack(tree)

    rasters = physical.gs.rasters
    assert physical.gs.groups == ("optical", "dem")
    assert rasters["optical"].red.chunks is not None
    assert rasters["optical"].red.values[0, 0] == pytest.approx(0.1)
    assert rasters["dem"].red.dtype == np.dtype("uint16")
    assert physical.gs.attrs.root.get(attrs.ACDD).title == "scene"


def test_a_stack_blanks_nodata_group_by_group_and_stays_lazy() -> None:
    tree = build_stack(
        {
            "optical": stored(0, scale=None, nodata=0).chunk(),
            "dem": stored(7, scale=None).chunk(),
        }
    ).gs.rebase(attrs.ACDD(title="scene"))

    blanked = to_nan(tree)

    rasters = blanked.gs.rasters
    assert rasters["optical"].red.chunks is not None
    assert np.isnan(rasters["optical"].red.values).all()
    assert rasters["dem"].red.dtype == np.dtype("uint16")
    assert blanked.gs.attrs.root.get(attrs.ACDD).title == "scene"
```

In `tests/geodata/transform/test_nodata.py`, append. The `stack` fixture comes from `tests/geodata/conftest.py`: two groups, `optical` holding `red` and `infrared` holding `nir`, on one 2x2 grid with values `[[1000, 2000], [3000, 0]]`.

```python
def test_masking_a_stack_keeps_its_root_attrs_and_stays_lazy(stack: xr.DataTree) -> None:
    titled = stack.chunk().gs.rebase(attrs.ACDD(title="scene"))
    valid = np.array([[True, False], [True, True]])

    masked = mask(titled, valid, fill=0)

    red = masked.gs.rasters["optical"].red
    assert red.chunks is not None
    assert red.values.tolist() == [[1000, 0], [3000, 0]]
    assert masked.gs.attrs.root.get(attrs.ACDD).title == "scene"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/transform/test_packing.py tests/geodata/transform/test_nodata.py -q`
Expected: the two packing tests FAIL, with `AttributeError` on a `DataTree` inside `unpack` and `to_nan`. The `mask` test may already PASS; it guards the rewrite in Step 3.

- [ ] **Step 3: Implement**

`src/geosave_engine/geodata/transform/packing.py`: replace the imports and `unpack` (everything above `_unpack_array`, below the module docstring) with:

```python
from __future__ import annotations

from typing import cast

import xarray as xr
from xarray.coding.variables import CFScaleOffsetCoder

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import nodata


def unpack[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Read physical values out of each variable's stored digital numbers.

    A variable carrying neither `scale_factor` nor `add_offset` passes through
    unchanged. A stack is unpacked group by group.

    Args:
        data: DataArray, Dataset, or DataTree holding stored values.

    Returns:
        New object of the same kind holding physical values, each unpacked
        variable's packing dropped from attrs since its values are no longer
        the stored numbers packing described.

    Raises:
        ValueError: A packed variable still marks its nodata pixels with a
            fill value, which scaling would read as an ordinary value.

    Examples:
        >>> unpack(scene).red.max().item()
        0.09
    """
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import map_groups

        return cast("T", map_groups(data, unpack))

    if isinstance(data, xr.DataArray):
        return cast("T", _unpack_array(data))

    physical = {
        variable: _unpack_array(data[variable]) for variable in data.gs.variables
    }
    return cast("T", data.assign(physical))
```

`src/geosave_engine/geodata/transform/nodata.py`: replace the two `@overload` stubs and `to_nan` with:

```python
def to_nan[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Replace each variable's fill value with NaN.

    A reduction reads a stored fill value as data unless it finds NaN there
    instead. A variable carrying no fill value passes through unchanged. A
    stack is blanked group by group.

    Args:
        data: DataArray, Dataset, or DataTree holding stored values.

    Returns:
        New object of the same kind holding NaN where the pixels were nodata.
        The fill value leaves attrs with them, since no pixel holds it now.

    Examples:
        >>> to_nan(scene).red.dtype
        dtype('float64')
        >>> to_nan(scene).red.attrs
        {}
    """
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import map_groups

        return cast("T", map_groups(data, to_nan))

    if isinstance(data, xr.DataArray):
        return cast("T", _to_nan_array(data))

    blanked = {
        variable: _to_nan_array(data[variable]) for variable in data.gs.variables
    }
    return cast("T", data.assign(blanked))
```

In the same file, replace the `DataTree` branch of `mask`:

```python
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import map_groups

        return cast(
            "T", map_groups(data, lambda raster: mask(raster, valid, fill=fill))
        )
```

Then remove what is now unused in both files: `overload` and `TYPE_CHECKING` from the `typing` import, and the `if TYPE_CHECKING:` import of `DataArray` and `Dataset`. Keep any name `uv run ruff check` does not report.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/transform -q`
Expected: all pass, 220 or more.

Run: `uv run ruff check src/geosave_engine/geodata/transform`
Expected: no findings.

---

### Task 3: `crop` on a stack, with one burner

**Files:**
- Modify: `src/geosave_engine/geodata/transform/vector.py` (the `crop` function)
- Test: `tests/geodata/transform/test_vector.py`

**Interfaces:**
- Consumes: `map_groups` from Task 1; `rasterize(vector, like, *, column=None, ...) -> xr.DataArray` in the same module, which returns a boolean presence mask on `like`'s grid; `GeoVector.footprint -> odc.geo.geom.Geometry`, which raises `ValueError` on an empty vector; `nodata.mask`.
- Produces: `crop[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T, vector: GeoVector, *, mask: bool = True) -> T`.

- [ ] **Step 1: Write the tests**

Append to `tests/geodata/transform/test_vector.py`:

```python
def _triangle():
    import geopandas as gpd
    import shapely

    from geosave_engine.geodata.core.vector import GeoVector

    return GeoVector(
        gpd.GeoDataFrame(
            geometry=[shapely.Polygon([(0, 0), (30, 0), (0, 30)])], crs="EPSG:32633"
        )
    )


def _band(times: int = 0):
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.array import array

    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    band = array(np.ones(grid.shape, "uint16"), grid, nodata=0).rename("red")
    if not times:
        return band
    return band.expand_dims(time=np.arange(times).astype("datetime64[D]")).copy()


def test_crop_masks_pixel_centres_inside_the_geometry() -> None:
    cropped = _band().gs.crop(_triangle())

    assert cropped.values.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_masks_every_instant_of_a_time_axis() -> None:
    cropped = _band(times=2).gs.crop(_triangle())

    assert cropped.shape == (2, 3, 3)
    assert cropped.values[0].tolist() == cropped.values[1].tolist()
    assert cropped.values[1].tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_cuts_every_group_of_a_stack_and_keeps_root_attrs() -> None:
    import geosave_engine.geodata.attrs as attrs
    from geosave_engine.geodata.core.stack import stack
    from geosave_engine.geodata.transform.vector import crop

    band = _band()
    tree = stack(
        {"optical": band.to_dataset(), "dem": band.rename("height").to_dataset()}
    ).gs.rebase(attrs.ACDD(title="scene"))

    cropped = crop(tree, _triangle())

    assert cropped.gs.groups == ("optical", "dem")
    assert cropped.gs.geobox.shape == (3, 3)
    assert cropped.gs.rasters["dem"].height.values.tolist() == [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
    assert cropped.gs.attrs.root.get(attrs.ACDD).title == "scene"


def test_crop_refuses_an_empty_vector() -> None:
    from geosave_engine.geodata.core.vector import GeoVector

    with pytest.raises(ValueError, match="empty GeoVector"):
        _band().gs.crop(GeoVector.empty("EPSG:32633"))
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/geodata/transform/test_vector.py -q`
Expected: the stack test FAILS with `AttributeError` on a `DataTree`, and the empty-vector test FAILS because today's error reads `cannot convert float NaN to integer`. The pixel-centre and time-axis tests PASS; they pin today's pixels so the burner swap cannot change them.

- [ ] **Step 3: Implement**

Replace the whole `crop` function in `src/geosave_engine/geodata/transform/vector.py` with:

```python
def crop[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, vector: GeoVector, *, mask: bool = True
) -> T:
    """Cut a raster, band, or stack down to a vector's extent.

    Args:
        data: Dataset or DataArray on a regular grid, or a stack of them,
            which is cut group by group.
        vector: Geometries to cut against, in `data`'s CRS.
        mask: Also make nodata the pixels outside the geometries, which then
            take their own variable's fill value.

    Returns:
        `data` covering the vector's extent, its dtype unchanged.

    Raises:
        ValueError: `data` carries no CRS, `vector` is empty, is in a
            different CRS or does not overlap `data`, the crop leaves no
            regular grid to mask on, or `mask` is set while a variable carries
            no fill value.

    Examples:
        >>> crop(scene, field_boundaries).gs.geobox.shape
        (64, 48)
    """
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import map_groups

        return cast(
            "T", map_groups(data, lambda raster: crop(raster, vector, mask=mask))
        )

    crs = data.odc.crs
    if crs is None:
        raise ValueError(
            f"{type(data).__name__} carries no CRS; write one with "
            f"gs.write_crs before cropping"
        )
    if vector.crs != crs:
        raise ValueError(
            f"vector is in {vector.crs} but the raster is in {crs}; "
            f"reproject the vector before cropping"
        )

    # odc's own apply_mask writes NaN, which promotes every integer variable.
    cut = cast("T", data.odc.crop(vector.footprint, apply_mask=False))
    if not mask:
        return cut
    if not isinstance(cut.odc.geobox, GeoBox):
        raise ValueError(
            f"crop left {type(cut).__name__} with no regular grid, so mask has "
            f"nothing to rasterize the vector onto"
        )
    return nodata.mask(cut, rasterize(vector, cut))
```

The CRS check runs before `vector.footprint`, so the existing "names the accessor" test keeps its message, and an empty vector then raises from `footprint`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/transform/test_vector.py tests/geodata/transform/test_nodata.py -q`
Expected: all pass, including the existing `test_cropping_with_a_mask_keeps_the_dtype_and_declared_fill`.

Run: `uv run ruff check src/geosave_engine/geodata/transform/vector.py`
Expected: no findings.

---

### Task 4: One accessor base

**Files:**
- Modify: `src/geosave_engine/geodata/core/base.py`
- Modify: `src/geosave_engine/geodata/core/array.py`
- Modify: `src/geosave_engine/geodata/core/raster.py:64`
- Modify: `src/geosave_engine/geodata/core/stack.py:35` and the `GeoStack` class line
- Test: `tests/geodata/core/test_stack.py`

**Interfaces:**
- Consumes: generic `packing.unpack`, `nodata.to_nan`, `nodata.mask`, `vector.crop` from Tasks 2 and 3; `warp.reproject`, already generic over a stack.
- Produces: `GeoRasterAccessor[DataT: xr.Dataset | xr.DataArray | xr.DataTree]` as the only class in `core/base.py`; `tensor` importable from `geosave_engine.geodata.core.array`. `GeoAccessor` no longer exists.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/core/test_stack.py`. The `stack` fixture is the two-group stack from `tests/geodata/conftest.py`.

```python
def test_a_stack_reads_the_pixel_operations_through_gs(stack: xr.DataTree) -> None:
    import geopandas as gpd
    import shapely

    from geosave_engine.geodata.core.vector import GeoVector

    valid = np.array([[True, False], [True, True]])
    left, bottom, right, top = stack.gs.bounds.bbox
    column = GeoVector(
        gpd.GeoDataFrame(
            geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
            crs=stack.gs.crs,
        )
    )

    assert stack.gs.unpack().gs.groups == stack.gs.groups
    assert stack.gs.to_nan().gs.groups == stack.gs.groups
    assert stack.gs.mask(valid, fill=0).gs.rasters["optical"].red.values[0, 1] == 0
    assert stack.gs.crop(column, mask=False).gs.geobox.shape == (2, 1)


def test_a_stack_reprojects_onto_a_target_rasters_own_grid(
    stack: xr.DataTree,
) -> None:
    target = build_raster(crs="EPSG:32748")

    warped = stack.gs.reproject(target)

    assert warped.gs.groups == stack.gs.groups
    assert warped.gs.geobox == target.gs.geobox
    assert stack.gs.reproject("EPSG:3857").gs.crs.epsg == 3857
```

`align` is not the same operation: its default `extent="union"` grows the grid to cover both the stack and the target, where `reproject` adopts the target's grid whole. Do not assert that the two agree.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_stack.py -q`
Expected: both FAIL with `AttributeError: 'GeoStack' object has no attribute 'unpack'` and `'reproject'`.

The `stack` fixture parameter shadows nothing here, since this file imports the builder as `build_stack`.

- [ ] **Step 3: Move `tensor`**

Cut the whole `tensor` function out of `src/geosave_engine/geodata/core/base.py` and paste it unchanged into `src/geosave_engine/geodata/core/array.py`, directly above `def array(`. In `array.py`:

```python
from .base import GeoRasterAccessor
```

and add `Callable` to the existing `collections.abc` import inside `if TYPE_CHECKING:`:

```python
    from collections.abc import Callable, Sequence
```

In `src/geosave_engine/geodata/core/raster.py`, replace line 64 with:

```python
from .array import tensor
from .base import GeoRasterAccessor
```

- [ ] **Step 4: Merge the classes**

In `src/geosave_engine/geodata/core/base.py`:

Replace the module docstring with:

```python
"""Properties and pixel operations shared by every `gs` raster accessor."""
```

Replace the `GeoAccessor` class statement and its docstring with:

```python
class GeoRasterAccessor[DataT: xr.Dataset | xr.DataArray | xr.DataTree]:
    """Spatial, attrs, and pixel members every `gs` raster accessor shares.

    Concrete accessors bind their xarray object to `_data`; this class is
    never built directly. Each pixel operation delegates to `transform`,
    which treats a band, a raster, and a stack alike.
    """
```

Delete the second class statement, `class GeoRasterAccessor[DataT: xr.Dataset | xr.DataArray](GeoAccessor[DataT]):`, and its docstring, so `unpack`, `mask`, `to_nan`, `reproject`, and `crop` become methods of the one class, after `anchor`.

Replace the two casting bodies:

```python
        return packing.unpack(self._data)
```

```python
        return nodata.to_nan(self._data)
```

Change the `typing` import to `from typing import TYPE_CHECKING, Any, Literal, overload`, and remove `Callable`, `torch`, and `DTypeLike` from the `if TYPE_CHECKING:` block.

In `src/geosave_engine/geodata/core/stack.py`, change the import and the class line:

```python
from .base import GeoRasterAccessor
```

```python
class GeoStack(GeoRasterAccessor["DataTree"]):
```

- [ ] **Step 5: Run the tests and the type checker**

Run: `uv run pytest tests/geodata -q`
Expected: all pass.

Run: `grep -rn "GeoAccessor\b" src tests`
Expected: no output.

Run: `grep -n "cast" src/geosave_engine/geodata/core/base.py`
Expected: no output.

Run: `uv run basedpyright src/geosave_engine/geodata/core`
Expected: `0 errors`.

Run: `uv run ruff check src/geosave_engine/geodata/core`
Expected: no findings.

---

### Task 5: Type errors and full verification

**Files:**
- Modify: `src/geosave_engine/geodata/transform/tiling.py:121-128`
- Modify: `src/geosave_engine/geodata/transform/time.py:313`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: no signature changes.

- [ ] **Step 1: See the two errors**

Run: `uv run basedpyright src/geosave_engine/geodata/transform`
Expected: `2 errors`, at `tiling.py:127` (`dict[Any, Dataset | DataArray]` is not `Mapping[str, Dataset]`) and `time.py:313` (`Series | Unknown | Self@DataFrame` is not `DataFrame`). If Task 2 or 3 introduced others, fix those in their own files first.

- [ ] **Step 2: Fix them**

In `src/geosave_engine/geodata/transform/tiling.py`, add `cast` to the `typing` import and replace the `DataTree` branch of `_map_groups`:

```python
    if isinstance(raster, xr.DataTree):
        # A group is a Dataset, and every change returns the kind it is given.
        return stack(
            {
                name: cast("xr.Dataset", change(group))
                for name, group in raster.gs.rasters.items()
            }
        )
```

In `src/geosave_engine/geodata/transform/time.py`, replace the last line of `_joint_axis`:

```python
    return table.loc[changed]
```

- [ ] **Step 3: Verify the whole change**

Run: `uv run basedpyright src/geosave_engine/geodata/core src/geosave_engine/geodata/transform`
Expected: `0 errors`.

Run: `uv run pytest -q`
Expected: all pass. A failure outside `tests/geodata` in code this plan did not touch is pre-existing work in progress; report it, do not fix it.

Run: `uv run ruff check src/geosave_engine/geodata tests/geodata`
Expected: no findings.

Run: `git diff --check`
Expected: no output.

- [ ] **Step 4: Report**

Report what changed, the checks run with their results, and any pre-existing failure found. Do not commit.
