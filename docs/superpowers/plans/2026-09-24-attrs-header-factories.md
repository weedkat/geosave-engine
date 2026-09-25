# Attrs Header Factories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace vague header readers with centralized `create_header` factories and simplify `rebase` without changing its convenient public forms.

**Architecture:** Context adapters under `geodata.attrs.headers` gather xarray, GeoBox, or STAC inputs and finish through `AttrsHeader.from_attrs`. `AttrsHeader` remains context-free, while `attrs.rebase` retains header replacement, namespace overlay, and ordered model-edit behavior with a direct implementation that no longer constructs a synthetic model patch.

**Tech Stack:** Python 3.12, xarray, odc-geo, pystac, odc-stac, Pydantic, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-24-attrs-header-factories-design.md`

## Global Constraints

- Keep `AttrsHeader` free of xarray, GeoBox, and STAC dependencies.
- Name every contextual factory `create_header`; expose the xarray factory as `attrs.create_header`.
- Keep `rebase` and all three existing input forms: `AttrsHeader`, `AttrsNamespace`, and ordered `AttrsModel` edits.
- Remove `read`, both `read_header` functions, and `_model_patch` without compatibility aliases.
- Preserve atomic in-place validation, shallow pixel copies, and independent mutable metadata on bulk edits.
- Preserve all unrelated working-tree changes.

## Review Focus

- A DataArray has no data-variable namespace: `create_header(array)` must put its own attrs at the root and still capture coordinate attrs.
- A DataTree factory call describes only the selected node, matching current behavior rather than traversing children.
- Importing `geosave_engine.geodata.attrs` must not eagerly import pystac or odc-stac through the specialized STAC factory.
- A later invalid rebase target must leave every earlier target unchanged during `inplace=True` writes.
- Reusing one mutable model value across several targets must not make their resulting attrs share the same object.

---

### Task 1: Centralize xarray and GeoBox header creation

**Files:**
- Create: `src/geosave_engine/geodata/attrs/headers/__init__.py`
- Create: `src/geosave_engine/geodata/attrs/headers/xarray.py`
- Create: `src/geosave_engine/geodata/attrs/headers/geobox.py`
- Modify: `src/geosave_engine/geodata/attrs/__init__.py`
- Modify: `src/geosave_engine/geodata/attrs/xarray.py`
- Modify: `src/geosave_engine/geodata/core/profile.py`
- Modify: `src/geosave_engine/geodata/core/array.py`
- Modify: `src/geosave_engine/geodata/core/raster.py`
- Modify: `src/geosave_engine/geodata/core/base.py`
- Modify: `src/geosave_engine/geodata/transform/composite.py`
- Modify: `src/geosave_engine/workflow/spec/requirements.py`
- Modify: `tests/geodata/attrs/test_header_stamping.py`
- Create: `tests/geodata/attrs/headers/test_xarray.py`
- Create: `tests/geodata/attrs/headers/test_geobox.py`

**Interfaces:**
- Consumes: `AttrsHeader.from_attrs(*, root, data_vars, coords) -> AttrsHeader`
- Produces: `attrs.create_header(obj: xr.Dataset | xr.DataArray | xr.DataTree) -> AttrsHeader`
- Produces: `attrs.headers.geobox.create_header(geobox: GeoBox) -> AttrsHeader`

- [ ] **Step 1: Write failing factory-interface tests**

Create focused tests that import the intended functions and pin the special object shapes:

```python
from geosave_engine.geodata import attrs
from geosave_engine.geodata.attrs.headers import geobox


def test_xarray_factory_captures_root_variables_and_coordinates() -> None:
    source = _source()
    header = attrs.create_header(source)
    assert header.root.to_attrs() == source.attrs
    assert header.data_vars["red"].to_attrs()["_FillValue"] == 0
    assert header.coords["x"].to_attrs() == source.x.attrs


def test_dataarray_factory_keeps_own_attrs_at_the_root() -> None:
    array = xr.DataArray([1], dims="x", attrs={"units": "1"})
    header = attrs.create_header(array)
    assert header.root.get(attrs.CFVariable).units == "1"
    assert header.data_vars == {}


def test_geobox_factory_describes_its_spatial_coordinates(utm_geobox) -> None:
    header = geobox.create_header(utm_geobox)
    assert header.coords["y"].to_attrs() == {
        "standard_name": "projection_y_coordinate",
        "units": "metre",
        "axis": "Y",
    }
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run pytest tests/geodata/attrs/headers/test_xarray.py tests/geodata/attrs/headers/test_geobox.py -q
```

Expected: collection or import failure because `attrs.headers` and `attrs.create_header` do not exist.

- [ ] **Step 3: Implement the factories and exports**

Implement the xarray adapter without importing specialized adapters eagerly:

```python
type XarrayObject = xr.Dataset | xr.DataArray | xr.DataTree


def create_header(obj: XarrayObject) -> AttrsHeader:
    variables = obj.coords.variables if isinstance(obj, xr.DataArray) else obj.variables
    data_vars = () if isinstance(obj, xr.DataArray) else obj.data_vars
    return AttrsHeader.from_attrs(
        root=obj.attrs,
        data_vars={str(name): variables[name].attrs for name in sorted(data_vars)},
        coords={str(name): variables[name].attrs for name in sorted(obj.coords)},
    )
```

Move the existing GeoBox-to-CF mapping into
`attrs/headers/geobox.py:create_header`. Keep
`attrs/headers/__init__.py` empty apart from its package docstring so importing
`attrs` does not import optional context dependencies. Re-export only the
xarray factory from `attrs/__init__.py`:

```python
from .headers.xarray import create_header
```

Remove `read` from `attrs/xarray.py`, update `flag_variables` and `merge` to
call `create_header`, and replace every direct `attrs.read(...)` call with
either `attrs.create_header(...)` or the existing `.gs.attrs` property.
Replace the former `core.convention.read_header` imports with the module-qualified
GeoBox factory and delete the old function.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/geodata/attrs/headers/test_xarray.py tests/geodata/attrs/headers/test_geobox.py tests/geodata/attrs/test_header_stamping.py tests/geodata/test_core_smoke.py -q
```

Expected: all selected tests pass.

### Task 2: Move STAC header construction into the factory package

**Files:**
- Create: `src/geosave_engine/geodata/attrs/headers/stac.py`
- Delete: `src/geosave_engine/geodata/stac/stamp.py`
- Modify: `src/geosave_engine/geodata/stac/source.py`
- Create: `tests/geodata/attrs/headers/test_stac.py`
- Delete: `tests/geodata/stac/test_header.py`
- Modify: `tests/geodata/stac/test_source.py`

**Interfaces:**
- Consumes: `AttrsHeader.from_attrs(*, root, data_vars, coords) -> AttrsHeader`
- Produces: `attrs.headers.stac.create_header(items, collection, loaded, *, groupby, item_properties=(), asset_fields=None, stac_cfg=None) -> AttrsHeader`
- Produces: `attrs.headers.stac.read_asset_fields(asset, *, band_index=1) -> dict[str, object]`

- [ ] **Step 1: Move tests to the intended import and add the lazy-import guard**

Move the existing STAC header behavior tests to the attrs header test tree and
change their import to:

```python
from geosave_engine.geodata.attrs.headers.stac import create_header
```

Rename calls from `read_header(...)` to `create_header(...)`. Add a subprocess
test so a future eager export cannot pull STAC dependencies into the common
attrs path:

```python
def test_importing_attrs_does_not_import_stac_adapters() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import geosave_engine.geodata.attrs; "
         "assert 'geosave_engine.geodata.attrs.headers.stac' not in sys.modules"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the moved tests and verify RED**

Run:

```bash
uv run pytest tests/geodata/attrs/headers/test_stac.py -q
```

Expected: import failure because `attrs.headers.stac` does not exist.

- [ ] **Step 3: Move the complete STAC translation implementation**

Move `StacGroupby`, `read_asset_fields`, STAC field parsing, conflict handling,
provenance assembly, and the header factory from `stac/stamp.py` into
`attrs/headers/stac.py`. Rename only `read_header` to `create_header`; keep
`read_asset_fields` because it reads one published STAC asset rather than
constructing a header. Import `AttrsHeader` relatively:

```python
from ..header import AttrsHeader
```

Update `stac/source.py` and its monkeypatch-based test to import and patch the
new factory location. Delete the obsolete implementation module rather than
leaving a forwarding adapter.

- [ ] **Step 4: Run STAC and acceptance tests and verify GREEN**

Run:

```bash
uv run pytest tests/geodata/attrs/headers/test_stac.py tests/geodata/stac/test_source.py tests/geodata/stac/test_dataflow.py -q
```

Expected: all selected tests pass with unchanged header values and loading behavior.

### Task 3: Simplify rebase while retaining its interface

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/xarray.py`
- Modify: `tests/geodata/attrs/test_header_stamping.py`
- Modify: repository callers and tests still importing `read` or `read_header`

**Interfaces:**
- Consumes: `rebase(obj, AttrsHeader | AttrsNamespace | *AttrsModel, target=None, inplace=False, **model_kwargs)`
- Produces: the same public overloads and behavior without `_model_patch`

- [ ] **Step 1: Strengthen the characterization test for clearing a keyword model**

Add a case that proves `None` clears every spelling declared by a model while
retaining unrelated attrs:

```python
def test_keyword_none_clears_every_key_owned_by_the_model() -> None:
    source = xr.Dataset(
        {"red": ("x", [1], {"_FillValue": 0, "nodata": 0, "kept": True})}
    )
    result = rebase(source, target="red", nodata=None)
    assert result.red.attrs == {"kept": True}
```

- [ ] **Step 2: Run the characterization tests before refactoring**

Run:

```bash
uv run pytest tests/geodata/attrs/test_header_stamping.py -q
```

Expected: PASS. This is the green starting point for a behavior-preserving refactor; the renamed factory tests supplied the failing interface tests in Tasks 1 and 2.

- [ ] **Step 3: Remove `_model_patch` and apply edits directly**

Keep header and namespace preparation unchanged. In the model branch, validate
all positional values as `AttrsModel`, serialize each positional model in
order, then process keyword models in keyword order. For a keyword value of
`None`, produce clears directly from `model_type.field_keys`; otherwise
construct and serialize the supplied model. Fold those ordered mappings into
one edit mapping, preserving the last-write-wins behavior:

```python
edits = [model.to_attrs() for model in positional_models]
for name, values in model_kwargs.items():
    model_type = resolve_model(name)
    edits.append(
        {
            key: None
            for keys in model_type.field_keys.values()
            for key in keys
        }
        if values is None
        else model_type(**values).to_attrs()
    )

patch = {}
for edit in edits:
    patch.update(edit)
```

Retain pre-resolution of every target before committing attrs. Deep-copy the
final non-`None` patch once per target so list and dict values remain
independent. Delete `_model_patch` and its now-unused `cast` for model sequences.

- [ ] **Step 4: Run focused behavioral tests**

Run:

```bash
uv run pytest tests/geodata/attrs tests/geodata/stac tests/geodata/test_core_smoke.py tests/geodata/transform/test_composite.py tests/geodata/test_raster_io_smoke.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Check imports, formatting, and the complete suite**

Run:

```bash
rg -n '\battrs\.read\b|\bread_header\b|\b_model_patch\b' src tests
uv run ruff check src/geosave_engine/geodata/attrs src/geosave_engine/geodata/core src/geosave_engine/geodata/stac src/geosave_engine/geodata/transform tests/geodata
uv run pytest
```

Expected: `rg` returns no obsolete symbol references, Ruff exits zero, and the full configured test suite passes. Any unrelated pre-existing suite failure is reported by exact test name and output rather than hidden.
