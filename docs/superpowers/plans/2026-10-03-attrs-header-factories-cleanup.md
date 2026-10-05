# Attrs Header Factories Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every header factory return a header `rebase` applies in one call, read single models directly, and let STAC item disagreements drop with a warning.

**Architecture:** `AttrsHeader.root` becomes optional, so a header names only what it describes. The grid factory builds complete coordinate attrs from odc's `xr_coords`. A shared `common_attrs` replaces the STAC agreement machinery and `to_array`'s inline intersection.

**Tech Stack:** Python 3.12, Pydantic 2, xarray, odc-geo, odc-stac, pytest

**Spec:** `docs/superpowers/specs/2026-10-03-attrs-header-factories-cleanup-design.md`

## Global Constraints

- `rebase` accepts exactly an `AttrsModel` sequence, one `AttrsNamespace`, or one `AttrsHeader`.
- STAC item disagreements never raise; they drop with `DroppedAttrsWarning`.
- No aliases for `_LABELLING`, `_PACKING`, `AssetConflict`, `_shared_fields`, or `_disagreement`.
- Name values for what they hold.
- No commits; work stays in the tree for the user's review.

## Review Focus

- A Dataset whose root holds a title keeps it through `write_crs`; Task 2 tests this.
- A 1×1 grid keeps its geobox after `write_crs`; Task 2 tests this.
- A STAC item missing an asset entirely drops that asset's fields with a warning instead of raising; Task 4 tests this.
- An empty band list never reaches `common_attrs`; `to_array` already refuses it upstream.
- NaN fill values agree with each other in `common_attrs`; Task 1 tests this.

---

### Task 1: `common_attrs`

**Files:** Modify `src/geosave_engine/geodata/attrs/model.py`, `attrs/__init__.py`, `core/raster.py` (`to_array`). Test `tests/geodata/attrs/test_field_values.py`.

- [ ] **Step 1: failing tests**

```python
def test_common_attrs_keeps_what_every_mapping_carries_alike():
    assert common_attrs(
        [{"units": "1", "nodata": np.nan, "a": 1}, {"units": "1", "nodata": float("nan")}]
    ) == {"units": "1", "nodata": pytest.approx(np.nan, nan_ok=True)}
    assert common_attrs([{"units": "1"}, {"units": "K"}]) == {}
```

- [ ] **Step 2:** run `uv run pytest tests/geodata/attrs/test_field_values.py -q`. Expected: ImportError.

- [ ] **Step 3: implement**

```python
def common_attrs(mappings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the keys every mapping carries with one value."""
    first, *rest = mappings
    return {
        key: value
        for key, value in first.items()
        if all(key in other and attrs_equal(other[key], value) for other in rest)
    }
```

Export it from `attrs`, and in `to_array` replace the inline `shared = {...}` with `shared = attrs.common_attrs(list(band_attrs.values()))`.

- [ ] **Step 4:** run `uv run pytest tests/geodata/attrs tests/geodata/core -q`. Expected: pass.

### Task 2: Partial headers and the complete grid header

**Files:** Modify `attrs/header.py`, `attrs/xarray.py` (`rebase`), `attrs/headers/geobox.py`, `core/raster.py` (`write_crs`), `core/array.py` (builder). Tests `tests/geodata/attrs/test_header_stamping.py`, `tests/geodata/attrs/headers/test_geobox.py`, `tests/geodata/core/test_raster.py`.

- [ ] **Step 1: failing tests**

```python
# test_header_stamping.py
def test_a_header_without_a_root_leaves_the_root_alone() -> None:
    ds = xr.Dataset({"red": ("x", [1])}, attrs={"title": "S2"})
    restored = rebase(ds, AttrsHeader.from_attrs(data_vars={"red": {"units": "1"}}))
    assert restored.attrs == {"title": "S2"}
    assert restored.red.attrs == {"units": "1"}

# test_geobox.py
def test_the_grid_header_carries_odc_and_cf_attrs() -> None:
    header = geobox.create_header(utm)
    assert header.root is None
    assert header.coords["x"].to_attrs() == {
        "units": "metre", "resolution": 10.0, "crs": "EPSG:32633",
        "standard_name": "projection_x_coordinate", "axis": "X",
    }

# test_raster.py
def test_write_crs_replaces_grid_coordinate_attrs_and_keeps_the_root() -> None:
    built = raster({"red": np.ones(geobox().shape)}, geobox())
    built.attrs = {"title": "S2"}
    built.x.attrs["crs"] = "EPSG:4326"
    written = built.gs.write_crs()
    assert written.attrs == {"title": "S2"}
    assert written.x.attrs["crs"] == "EPSG:32633"
    assert written.x.attrs["axis"] == "X"
```

Add a 1×1 grid case asserting `written.odc.geobox == source.odc.geobox`.

- [ ] **Step 2:** run them. Expected: the root test fails (root replaced), the geobox test fails (no odc keys), and the raster test fails (stale `crs`).

- [ ] **Step 3: implement**
  - `AttrsHeader.root: AttrsNamespace | None = None`.
  - `from_attrs(root=None)` leaves it out.
  - `merge` merges only the roots present, and the result has no root when none is.
  - `rebase` adds the root update only when `header.root is not None`.
  - `geobox.create_header` follows the spec code.
  - `write_crs` and the array builder call `attrs.rebase(result, create_geobox_header(geobox))`.

- [ ] **Step 4:** run `uv run pytest tests/geodata -q`. Expected: pass.

### Task 3: Single-model reads

**Files:** `utils/io/zarr.py` (order, fill encoding), `core/base.py` (`timespan`), `transform/nodata.py` (2 sites), `transform/packing.py`.

- [ ] **Step 1:** replace each read with `Model.from_attrs(<mapping>)`, as the spec table shows. This changes no behaviour, so the existing suites are the guard. Run `uv run pytest tests/geodata/utils/io tests/geodata/transform tests/geodata/core -q` before and after. Expected: identical pass counts.

### Task 4: STAC disagreements drop with a warning

**Files:** `attrs/headers/stac.py`, `stac/source.py`. Test `tests/geodata/attrs/headers/test_stac.py`.

- [ ] **Step 1: failing tests**

Replace `test_reading_rejects_packing_missing_from_one_item` and `test_reading_rejects_mixed_packing_values` with:

```python
@pytest.mark.parametrize("second", [{}, {"scale": 0.0002}])
def test_disagreeing_packing_drops_with_a_warning(second) -> None:
    items = [_item("scene-1", dt(2025, 1, 1), {"scale": 0.0001}), _item("scene-2", dt(2025, 1, 2), second)]
    with pytest.warns(DroppedAttrsWarning, match="scale_factor") as warned:
        header = create_header(items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id")
    assert header.data_vars["red"].get(Packing) is None
    assert "scene-2" in str(warned[0].message)
```

Make `test_reading_drops_a_label_missing_from_one_item` expect the warning too. Add a test where an item lacks the asset entirely: it warns and drops.

- [ ] **Step 2:** run. Expected: the packing cases raise `ValueError` and the label case warns nothing.

- [ ] **Step 3: implement.** Use `_ATTR_KEYS` and `common_attrs` as the spec shows. The warning names the variable, the dropped keys, and `{item.id: value}` per key. Delete `_LABELLING`, `_PACKING`, `AssetConflict`, `_shared_fields`, and `_disagreement`, and move `StacGroupby` to `stac/source.py`.

- [ ] **Step 4:** run `uv run pytest tests/geodata/attrs/headers/test_stac.py tests/geodata/stac -q`. Expected: pass.

### Task 5: Verification

- [ ] Run `uv run pytest -q` (all pass), `uv run ruff check src tests` (clean), and the docstring checker on touched files (clean, apart from known pre-existing findings).
