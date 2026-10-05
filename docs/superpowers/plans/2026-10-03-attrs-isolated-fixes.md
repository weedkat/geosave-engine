# Attrs Isolated Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three isolated attrs defects: a dead `GDALVariable.description` field that captures foreign attrs, STAC loads that silently drop packing published by only some items, and a `rebase` docstring that misstates the header contract.

**Architecture:** Each task changes one function or field in `geodata.attrs` and pins the behaviour with a test in the mirrored test file. No public signature changes, no new module, no new dependency.

**Tech Stack:** Python 3.12, Pydantic 2, xarray, pystac, odc-stac, pytest

**Spec:** `docs/superpowers/specs/2026-10-03-geodata-attrs-review-design.md` (plan A)

## Global Constraints

- Packing fields (`scale`, `offset`) must agree in presence and value across every item loaded into one variable; labelling fields (`unit`, `description`) keep their drop-on-disagreement behaviour.
- `rebase(obj, header)` replaces the root and every variable or coordinate the header names; unnamed ones keep their attrs.
- A `description` attr is foreign after this plan; no model owns it.
- Do not add compatibility aliases for the removed field.
- The working tree holds unrelated staged and unstaged user work: commit only the listed paths with `git commit -m ... -- <paths>`, never `git add -A`, `git checkout --`, or `git stash`.

## Review Focus

- A STAC search spanning a processing-baseline change, such as Sentinel-2 L2A across January 2022, may now raise where it used to load. That is the intended fail-fast behaviour, and the error must name the item publishing each value; Task 2 asserts the item ids appear in the message.
- Items that all omit `scale` and `offset` must still load with no `Packing`; Task 2 pins this.
- A label field present on only some items must still be dropped without raising; Task 2 pins this.
- A GeoTIFF band carrying a `description` metadata tag must round-trip as a foreign attr; Task 1 runs the GeoTIFF suite.
- A header restore onto an object with extra variables must leave their attrs untouched; Task 3 pins this.

---

### Task 1: Remove the dead `GDALVariable.description` field

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/models/gdal.py:46`
- Test: `tests/geodata/attrs/test_gdal_variable.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GDALVariable` with fields `variable_name` and `colorinterp` only.

- [ ] **Step 1: Write the failing test**

Change the import line of `tests/geodata/attrs/test_gdal_variable.py` to:

```python
from geosave_engine.geodata.attrs import AttrsNamespace, GDALVariable, rebase
```

Append:

```python
def test_a_description_attr_stays_foreign() -> None:
    namespace = AttrsNamespace.from_attrs({"description": "Red band"})

    assert namespace.get(GDALVariable) is None
    assert namespace.foreign == {"description": "Red band"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/geodata/attrs/test_gdal_variable.py::test_a_description_attr_stays_foreign -v`
Expected: FAIL, `assert GDALVariable(variable_name=None, colorinterp=None, description='Red band') is None`.

- [ ] **Step 3: Delete the field**

In `src/geosave_engine/geodata/attrs/models/gdal.py`, delete this line:

```python
    description: Annotated[str, Field(min_length=1)] | None = None
```

`Annotated` and `Field` stay imported; `variable_name` still uses them.

- [ ] **Step 4: Run the attrs and raster I/O suites**

Run: `uv run pytest tests/geodata/attrs tests/geodata/utils/io -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "fix: leave description attrs foreign instead of GDALVariable" -- \
  src/geosave_engine/geodata/attrs/models/gdal.py \
  tests/geodata/attrs/test_gdal_variable.py
```

---

### Task 2: Refuse packing published by only some STAC items

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/headers/stac.py:170-171`
- Test: `tests/geodata/attrs/headers/test_stac.py:77-87`

**Interfaces:**
- Consumes: nothing.
- Produces: `create_header(...)` raising `ValueError` when packing fields differ in presence across items; signature unchanged.

- [ ] **Step 1: Replace the test that pinned the silent drop**

Add `Packing` to the attrs import in `tests/geodata/attrs/headers/test_stac.py`:

```python
from geosave_engine.geodata.attrs import CFVariable, Nodata, Packing
```

Replace the whole `test_reading_skips_field_missing_from_one_item` function with:

```python
def test_reading_rejects_packing_missing_from_one_item() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"scale": 0.0001}),
        _item("scene-2", dt(2025, 1, 2), {}),
    ]

    with pytest.raises(ValueError, match="'scale'") as raised:
        create_header(
            items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
        )

    assert "'scene-1' publishes 0.0001" in str(raised.value)
    assert "'scene-2' publishes None" in str(raised.value)


def test_reading_writes_no_packing_when_no_item_publishes_it() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {"unit": "1"}),
    ]

    header = create_header(
        items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(Packing) is None


def test_reading_drops_a_label_missing_from_one_item() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {}),
    ]

    header = create_header(
        items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(CFVariable) is None
```

- [ ] **Step 2: Run the tests to verify the reject case fails**

Run: `uv run pytest tests/geodata/attrs/headers/test_stac.py -v`
Expected: `test_reading_rejects_packing_missing_from_one_item` FAILS with `DID NOT RAISE`; the other two new tests pass, because they pin behaviour that must not change.

- [ ] **Step 3: Count absence as a value**

In `_shared_fields` in `src/geosave_engine/geodata/attrs/headers/stac.py`, replace:

```python
        per_item = [entry.get(key) for entry in published]
        if any(value is None for value in per_item):
            continue
```

with:

```python
        per_item = [entry.get(key) for entry in published]
        if all(value is None for value in per_item):
            continue
```

A field some items publish and others omit now reaches the disagreement check, so `"drop"` still skips it and `"reject"` raises.

- [ ] **Step 4: Run the STAC suites**

Run: `uv run pytest tests/geodata/attrs/headers/test_stac.py tests/geodata/stac -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "fix: refuse STAC packing published by only some items" -- \
  src/geosave_engine/geodata/attrs/headers/stac.py \
  tests/geodata/attrs/headers/test_stac.py
```

---

### Task 3: State the header-restore contract in `rebase`

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/xarray.py:224-240`
- Test: `tests/geodata/attrs/test_header_stamping.py`

**Interfaces:**
- Consumes: nothing.
- Produces: no behaviour change; `rebase` docstring and a test state the contract.

- [ ] **Step 1: Write the pinning test**

Append to `tests/geodata/attrs/test_header_stamping.py`:

```python
def test_rebase_header_leaves_variables_it_does_not_name() -> None:
    header = AttrsHeader.from_attrs(data_vars={"red": {"units": "1"}})
    target = xr.Dataset(
        {"red": ("x", [1], {"stale": True}), "nir": ("x", [2], {"units": "1"})}
    )

    restored = rebase(target, header)

    assert restored.red.attrs == {"units": "1"}
    assert restored.nir.attrs == {"units": "1"}
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/geodata/attrs/test_header_stamping.py::test_rebase_header_leaves_variables_it_does_not_name -v`
Expected: PASS. The behaviour already matches the contract; this test keeps it from drifting.

- [ ] **Step 3: Correct the docstring**

In `rebase` in `src/geosave_engine/geodata/attrs/xarray.py`, replace:

```python
    A header replaces the object's attrs whole; a namespace or bare models
    instead patch just `target`, leaving its other keys alone. The complete
    request is validated before any attrs change, including inplace writes.
```

with:

```python
    A header replaces the root attrs and those of every variable or coordinate
    it names; a namespace or bare models patch just `target`. The complete
    request is validated before any attrs change, including inplace writes.
```

and replace:

```python
        *models: Exactly one `AttrsHeader` to restore whole, since it is
            already a full snapshot naming its own variables; exactly one
```

with:

```python
        *models: Exactly one `AttrsHeader` to restore the root and every
            variable or coordinate it names; exactly one
```

- [ ] **Step 4: Run the attrs suite and lint**

Run: `uv run pytest tests/geodata/attrs -q && uv run ruff check src/geosave_engine/geodata/attrs tests/geodata/attrs && uv run python scripts/check_docstrings.py src/geosave_engine/geodata/attrs/xarray.py`
Expected: all pass, no ruff findings, no docstring findings.

- [ ] **Step 5: Commit**

```bash
git commit -m "docs: state which attrs a header rebase replaces" -- \
  src/geosave_engine/geodata/attrs/xarray.py \
  tests/geodata/attrs/test_header_stamping.py
```

---

### Task 4: Full verification

**Files:** none changed.

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. Baseline on 2026-10-03 before Task 1 was `1075 passed, 10 deselected`; this plan adds three tests and removes one, so expect `1077 passed, 10 deselected`.

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`
Expected: no new findings in files this plan touched.
