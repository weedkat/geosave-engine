# GeoTIFF Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make attrs collection parsing explicit, make one-field parsing side-effect free, retain stacked variable metadata for GeoTIFF, and derive truthful TIFF tags at persistence time.

**Architecture:** Shared field conversion lives in `attrs/validate.py`; concrete models retain their domain validators. `StackedAttrs` records the information xarray cannot represent after Dataset variables are stacked. `GeoTIFFTags.from_xarray()` synchronizes redundant TIFF tags immediately before the persistence adapter writes each scene.

**Tech Stack:** Python 3.12, xarray, Pydantic 2, odc-geo, rasterio, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-geotiff-metadata-design.md`

## Global Constraints

- Preserve unrelated changes in the existing dirty worktree.
- Do not add compatibility aliases during Alpha development.
- Do not add dependencies.
- Keep generic `rebase()` free of GeoTIFF-specific synchronization.
- Write behavior tests before production changes and observe the expected failure.
- Do not commit automatically from the shared dirty worktree.

## Review Focus

- A JSON-looking unknown attr remains text because its intended type is unknown; Task 1 keeps decoding field-scoped.
- One-field parsing retains `Annotated` constraints but does not run model-level validators; Task 1 covers both behaviors.
- Selecting a subset of stacked bands restores only those variables; Task 2 retains the existing regression.
- An explicit carried TIFF description survives when ACDD supplies no summary; Task 3 adds this case.
- Map scale never turns degrees into physical resolution; Task 3 rejects geographic grids.

---

### Task 1: Shared collection validation and one-field parsing

**Files:**
- Create: `src/geosave_engine/geodata/attrs/validate.py`
- Modify: `src/geosave_engine/geodata/attrs/model.py`
- Modify: `src/geosave_engine/geodata/attrs/namespace.py`
- Modify: `src/geosave_engine/geodata/attrs/__init__.py`
- Modify: `src/geosave_engine/geodata/attrs/models/legend.py`
- Modify: `src/geosave_engine/geodata/attrs/models/stac.py`
- Modify: `src/geosave_engine/geodata/attrs/models/zarr.py`
- Modify: `src/geosave_engine/workflow/spec/requirements.py`
- Test: `tests/geodata/attrs/test_field_values.py`
- Test: `tests/geodata/attrs/test_namespace.py`

**Interfaces:**
- Produces: `parse_collection_text(value: object) -> object`.
- Produces: `parse_field_value(model: type[AttrsModel], field: str, value: object) -> Any`.
- Produces: `AttrsModel._field_parsers`, built once at subclass registration.

- [ ] **Step 1: Write failing tests for the new parsing interface**

Rename the existing `read_field` tests to call `parse_field_value`, add a model whose decorator validator proves single-field parsing does not construct the model, and assert `parse_collection_text` accepts native collections, decodes JSON collection text, and leaves invalid text for field validation.

- [ ] **Step 2: Verify the tests fail because the new interfaces do not exist**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/geodata/attrs/test_field_values.py tests/geodata/attrs/test_namespace.py`

Expected: collection fails because `attrs.validate` is absent and field parsing fails because `parse_field_value` is absent.

- [ ] **Step 3: Implement the shared validator and eager field parsers**

Create:

```python
def parse_collection_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return orjson.loads(value)
    except orjson.JSONDecodeError:
        return value
```

At `AttrsModel` subclass registration, build one immutable mapping of field name to `TypeAdapter(field.rebuild_annotation())`. Implement `parse_field_value` as lookup plus `validate_python`, with a concise docstring explaining why parsing one field is necessary. Remove `_FIELD_ADAPTERS`, `Collected`, `from_json_text`, and `read_field`. Update every consumer and export without aliases.

- [ ] **Step 4: Replace collection aliases with explicit annotations**

Use this shape on Legend, STAC, and Zarr collection fields:

```python
flag_values: Annotated[
    list[int] | None,
    BeforeValidator(parse_collection_text),
] = None
```

Leave `_spell_class_map`, pair validation, GeoTIFF parsing, colour interpretation, and TimeSpec validation in their concrete models.

- [ ] **Step 5: Run focused attrs tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/geodata/attrs`

Expected: all attrs tests pass.

### Task 2: Rename and retain stacked metadata

**Files:**
- Create: `src/geosave_engine/geodata/attrs/models/stacked.py`
- Delete: `src/geosave_engine/geodata/attrs/models/band.py`
- Modify: `src/geosave_engine/geodata/attrs/models/__init__.py`
- Modify: `src/geosave_engine/geodata/attrs/__init__.py`
- Modify: `src/geosave_engine/geodata/core/raster.py`
- Modify: `src/geosave_engine/geodata/core/array.py`
- Test: `tests/geodata/test_raster.py`
- Test: `tests/geodata/test_raster_io_smoke.py`

**Interfaces:**
- Consumes: `parse_collection_text` from Task 1.
- Produces: `StackedAttrs.from_variables(...)`, `shared()`, `restore()`, and `restore_root()`.

- [ ] **Step 1: Write failing conversion and GeoTIFF tests**

Update conversion tests to retrieve `StackedAttrs`. Add a real GeoTIFF round trip in which red and blue variables have different `units`, `scale_factor`, and `colorinterp`; call `source.gs.to_array().gs.to_cog(path)` and assert rasterio/GDAL reads the distinct band metadata back.

- [ ] **Step 2: Verify failure on the missing renamed model**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/geodata/test_raster.py tests/geodata/test_raster_io_smoke.py`

Expected: collection fails because `StackedAttrs` is absent.

- [ ] **Step 3: Rename the model without a compatibility alias**

Move the implementation to `models/stacked.py`, rename fields to `variable_attrs` and `dataset_attrs`, and apply explicit `BeforeValidator(parse_collection_text)` annotations. Update core references and docstrings. Keep selected-band and shadowed-root restoration unchanged.

- [ ] **Step 4: Run raster conversion and persistence tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/geodata/test_raster.py tests/geodata/test_raster_io_smoke.py`

Expected: all tests pass.

### Task 3: Synchronize GeoTIFF tags from xarray metadata

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/models/geotiff.py`
- Modify: `src/geosave_engine/geodata/utils/io/geotiff.py`
- Modify: `src/geosave_engine/geodata/utils/io/layout.py`
- Modify: `src/geosave_engine/geodata/core/array.py`
- Modify: `src/geosave_engine/geodata/core/raster.py`
- Modify: `src/geosave_engine/geodata/core/stack.py`
- Test: `tests/geodata/attrs/test_field_values.py`
- Test: `tests/geodata/test_raster_io_smoke.py`
- Test: `tests/geodata/test_layout_smoke.py`

**Interfaces:**
- Produces: `GeoTIFFTags.from_xarray(obj: xr.Dataset | xr.DataArray, *, map_scale: float | None = None) -> GeoTIFFTags`.
- Produces: explicit `map_scale: float | None` on COG, GTiff, layout, raster, array, and stack writers.

- [ ] **Step 1: Write failing synchronization tests**

Use literal expectations to cover: ACDD summary to image description; scalar time overriding a stale tag; carried description retained without ACDD summary; 10 m pixels at 1:10,000 producing 10 pixels/cm; US survey feet converted by the CRS axis factor; invalid scales rejected; geographic and missing grids rejected only when `map_scale` is supplied.

- [ ] **Step 2: Write a failing time-cube propagation test**

Write two instants with `map_scale=10_000`, inspect every COG leaf using rasterio, and assert resolution unit `3` with the expected X/Y values plus intact band metadata and time on readback.

- [ ] **Step 3: Verify tests fail because synchronization and map scale are absent**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/geodata/test_raster_io_smoke.py tests/geodata/test_layout_smoke.py`

Expected: failures name missing `from_xarray`/`map_scale` behavior.

- [ ] **Step 4: Implement `GeoTIFFTags.from_xarray`**

Read current root tags and ACDD from the object's attrs, update authoritative duplicates, and calculate each projected axis as:

```python
pixels_per_cm = 0.01 * map_scale / (
    abs(grid.resolution.axis) * crs_axis.unit_conversion_factor
)
```

Require a positive finite scale, a regular grid, a projected CRS, and two positive finite linear conversion factors. Set `TIFFTAG_RESOLUTIONUNIT=3` only after successful calculation.

- [ ] **Step 5: Centralize synchronization in GeoTIFF persistence**

After squeezing one time step, call `GeoTIFFTags.from_xarray(cube, map_scale=map_scale)`, then drop the scalar time coordinate and write the synchronized header. Thread `map_scale` explicitly through all public write interfaces and every time-cube leaf.

- [ ] **Step 6: Run focused persistence tests and lint**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/geodata/attrs tests/geodata/test_raster.py tests/geodata/test_raster_io_smoke.py tests/geodata/test_layout_smoke.py`

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/geodata/attrs src/geosave_engine/geodata/core/array.py src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/stack.py src/geosave_engine/geodata/utils/io tests/geodata/attrs tests/geodata/test_raster.py tests/geodata/test_raster_io_smoke.py tests/geodata/test_layout_smoke.py`

Expected: tests and Ruff pass.

### Task 4: Final verification

**Files:** No new production files.

**Interfaces:** Verifies all preceding interfaces together.

- [ ] **Step 1: Run the full suite**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest`

Expected: all tests pass, or sandbox-sensitive persistence failures are reported by exact test name.

- [ ] **Step 2: Run repository checks**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check .`

Run: `git diff --check`

Expected: both commands pass.

- [ ] **Step 3: Review the complete diff against the spec**

Confirm no compatibility aliases, generic rebase hooks, duplicated GeoTIFF synchronization, unrelated edits, or cache mutation remain.

