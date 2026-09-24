# GeoVector Spatial Collection Simplification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify GeoVector into a readable spatial collection whose xarray metadata comes from native accessors, whose identity is caller-owned, and whose raster conversion mechanics live in the transform package.

**Architecture:** DataArray, Dataset, and DataTree expose ordered variable identities through the same `.gs.variables` property. `GeoVector` keeps construction, composition, querying, and thin persistence/conversion methods; `geodata.transform.vector` owns eager Rasterio conversion. GeoParquet covering bboxes remain an explicit GeoPandas option and asset references are local paths.

**Tech Stack:** Python 3.12+, GeoPandas, pandas, Shapely, xarray, Dask, ODC Geo, Rasterio, PyArrow, pytest, Ruff, BasedPyright, mypy.

**Spec:** `docs/superpowers/specs/2026-09-24-geovector-spatial-collection-design.md`

## Global Constraints

- Preserve native GeoDataFrame, xarray, ODC, and Rasterio objects in public APIs.
- Do not infer identity from geometry, time, or an anchor; keyed replacement consumes a caller-owned column.
- Registration of xarray metadata must not compute pixels; vectorize and rasterize are explicit eager boundaries.
- Use `Literal` aliases for selectable fields and spatial predicates, explicit unions for query targets, and `object` for caller-owned property values.
- Do not use a private helper for one call site; keep a short operation inline with a domain comment or move a reusable responsibility to its owning module.
- Covering-bbox persistence is opt-in through GeoPandas' native `write_covering_bbox` option.
- Catalog asset paths are local, remain below the catalog directory after symlink resolution, and are stored relative to it.
- Preserve all unrelated staged and unstaged work. Target implementation paths were already dirty before this revision, so do not commit them.

## Review Focus

- An unnamed DataArray and a multi-group DataTree must produce unambiguous ordered `.gs.variables` values without reading pixels.
- `fields=()` must produce geometry plus only explicitly supplied properties; no identity column may appear.
- A dynamic unsupported predicate or metadata field must fail clearly even though the public parameter is Literal-typed.
- Default GeoParquet output must not contain a covering bbox; explicit opt-in must support bbox-filtered reads.
- Numeric property values must round-trip through the requested raster dtype exactly, and non-numeric properties must fail at the GeoSave boundary.

---

## File map

- `src/geosave_engine/geodata/core/array.py`: DataArray variable identity.
- `src/geosave_engine/geodata/core/raster.py`: existing Dataset variable identity, unchanged in behavior.
- `src/geosave_engine/geodata/core/stack.py`: group-qualified DataTree variable identity.
- `src/geosave_engine/geodata/core/vector.py`: lean collection state, constructors, composition, query, persistence delegation, and thin transform methods.
- `src/geosave_engine/geodata/transform/vector.py`: eager polygonization and rasterization.
- `src/geosave_engine/geodata/utils/io/geoparquet.py`: local portable paths and atomic writes.
- `src/geosave_engine/geodata/utils/io/__init__.py`: local vector source context.
- `tests/geodata/test_core_smoke.py`: common accessor variable contract.
- `tests/geodata/test_vector.py`: collection, constructor, path, and transform behavior.
- `tests/geodata/test_raster_io_smoke.py`: explicit covering-bbox persistence.

### Task 1: Put variable identity on every xarray accessor

**Files:**
- Modify: `tests/geodata/test_core_smoke.py`
- Modify: `src/geosave_engine/geodata/core/array.py`
- Modify: `src/geosave_engine/geodata/core/stack.py`

**Interfaces:**
- Consumes: native `DataArray.name`, `Dataset.gs.variables`, and `DataTree.gs.rasters`.
- Produces: `GeoArray.variables -> tuple[str, ...]` and `GeoStack.variables -> tuple[str, ...]`, consumed by Task 2.

- [ ] **Step 1: Add failing common-accessor tests**

```python
def test_every_xarray_accessor_names_its_variables(
    raster: xr.Dataset, stack: xr.DataTree
) -> None:
    assert raster.red.gs.variables == ("red",)
    assert raster.gs.variables == ("red", "nir")
    assert stack.gs.variables == ("optical/red", "infrared/nir")


def test_unnamed_array_has_no_variable_identity(raster: xr.Dataset) -> None:
    assert raster.red.rename(None).gs.variables == ()
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/geodata/test_core_smoke.py -q`

Expected: both new tests fail because `GeoArray` and `GeoStack` lack `variables`.

- [ ] **Step 3: Implement the two native properties**

```python
@property
def variables(self) -> tuple[str, ...]:
    """Name this array as a data variable, when it has a name."""
    return () if self._data.name is None else (str(self._data.name),)
```

```python
@property
def variables(self) -> tuple[str, ...]:
    """Name every grouped data variable in stack order."""
    return tuple(
        f"{group}/{variable}"
        for group, raster in self.rasters.items()
        for variable in raster.gs.variables
    )
```

- [ ] **Step 4: Verify GREEN and typing**

Run: `uv run pytest tests/geodata/test_core_smoke.py -q`

Expected: all tests pass.

Run: `uv run basedpyright src/geosave_engine/geodata/core/array.py src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/stack.py tests/geodata/test_core_smoke.py`

Expected: `0 errors`.

- [ ] **Step 5: Record the checkpoint without committing dirty paths**

Record Task 1 as complete in the execution ledger. Do not stage or commit the accessor or test files because they contained unrelated work before this plan.

### Task 2: Simplify GeoVector construction, composition, query, and paths

**Files:**
- Modify: `tests/geodata/test_vector.py`
- Modify: `tests/geodata/test_raster_io_smoke.py`
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `src/geosave_engine/geodata/utils/io/geoparquet.py`
- Modify: `src/geosave_engine/geodata/utils/io/__init__.py`

**Interfaces:**
- Consumes: `.gs.variables` from Task 1 and existing `.gs.anchor`.
- Produces: `AnchorField`, `XarrayField`, `SpatialPredicate`, simplified constructors, caller-keyed `upsert`, local path resolution, and explicit covering-bbox persistence.

- [ ] **Step 1: Rewrite tests around caller-owned identity**

Remove generated `anchor_id` assertions and pin the minimal record and caller-keyed replacement:

```python
def test_xarray_fields_are_explicit() -> None:
    vector = GeoVector.from_xarray(build_raster(), fields=())
    assert set(vector.gdf) == {"geometry"}


def test_upsert_uses_a_caller_owned_key() -> None:
    old = GeoVector.from_geometry(Point(0, 0), record_id="a", status="old")
    replacement = GeoVector.from_geometry(
        Point(1, 1), record_id="a", status="complete"
    )
    result = old.upsert(replacement, on="record_id")
    assert list(result.gdf.status) == ["complete"]
```

Use `typing.cast` only in runtime rejection tests so deliberately invalid Literal inputs do not create type-check failures:

```python
with pytest.raises(ValueError, match="predicate"):
    vector.query(Point(0, 0), predicate=cast("SpatialPredicate", "touch-ish"))
```

Change filtered GeoParquet reads to opt in explicitly and test the default schema:

```python
path = vector.to_geoparquet(
    tmp_path / "catalog.parquet", write_covering_bbox=True
)


def test_geoparquet_does_not_write_covering_bbox_by_default(tmp_path: Path) -> None:
    path = GeoVector.from_geometry(Point(0, 0)).to_geoparquet(
        tmp_path / "catalog.parquet"
    )
    assert "bbox" not in pq.read_schema(path).names
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py -q`

Expected: failures show the generated `anchor_id`, implicit covering bbox, and old upsert assumptions.

- [ ] **Step 3: Replace helper-heavy construction with direct typed blocks**

At module scope define only public vocabulary:

```python
type AnchorField = Literal["time", "grid"]
type XarrayField = AnchorField | Literal["variables"]
type SpatialPredicate = Literal[
    "intersects", "within", "contains", "covers", "covered_by"
]
```

Use `**properties: object` and `Mapping[str, object]`. Remove hashing, `_anchor_id`, `_utc`, `_checked_fields`, `_variable_names`, `_canonical_frame`, and `_query_geometry`.

In each constructor, validate its field set directly, build one `derived` dictionary, reject `properties.keys() & derived.keys()`, then call `from_geometry`. `from_xarray` obtains variable names only from `data.gs.variables`.

In `concat`, normalize active geometry names inside the single loop that builds frames. In `query`, resolve the explicitly typed target union in one commented block before transforming its footprint. Keep runtime membership validation for dynamic callers while typing `predicate: SpatialPredicate`.

- [ ] **Step 4: Make covering bboxes native and opt-in**

Restore `GeoVector.to_geoparquet` to a direct typed delegation:

```python
return geoparquet.write(
    geoparquet.portable_paths(self.gdf, path),
    path,
    overwrite=overwrite,
    **options,
)
```

Do not mutate or reconstruct `options`. Keep `write_covering_bbox` in the existing native `GeoParquetWriteOptions` TypedDict.

Remove URI string branches from `portable_paths`, `resolve_path`, and `read_vector`. Validate non-null path-column values as `str | PathLike[str]`, resolve them locally, reject escapes, and store relative POSIX text. Set a read vector's source with `Path(source).resolve()`.

- [ ] **Step 5: Verify GREEN and typing**

Run: `uv run pytest tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py -q`

Expected: all tests pass.

Run: `uv run basedpyright src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py`

Expected: no GeoVector-change errors. If a touched test file exposes unrelated pre-existing diagnostics, record their exact locations separately and keep every changed line type-clean.

- [ ] **Step 6: Record the checkpoint without committing dirty paths**

Record Task 2 as complete in the ledger. Do not stage or commit these pre-dirty implementation files.

### Task 3: Move raster/vector conversion into the transform package

**Files:**
- Create: `src/geosave_engine/geodata/transform/vector.py`
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `tests/geodata/test_vector.py`

**Interfaces:**
- Consumes: `GeoVector`, `GeoAnchor`, native `.gs.geobox`, `nodata.fill_value`, `nodata.check_fill_fits`, Rasterio `shapes` and `rasterize`.
- Produces: `vectorize(flags, *, value_name, mask, connectivity) -> GeoVector` and `rasterize(vector, like, *, column, fill, dtype, all_touched) -> xr.DataArray`; GeoVector methods delegate to these functions.

- [ ] **Step 1: Add failing ownership tests**

```python
from geosave_engine.geodata.transform.vector import rasterize, vectorize


def test_vector_method_delegates_to_vector_transform(raster) -> None:
    flags = raster.red.astype("uint8")
    direct = vectorize(flags)
    through_method = GeoVector.vectorize(flags)
    assert direct.gdf.equals(through_method.gdf)


def test_raster_method_delegates_to_vector_transform(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.anchor.geobox.extent)
    direct = rasterize(vector, raster)
    through_method = vector.rasterize(raster)
    xr.testing.assert_identical(direct, through_method)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/geodata/test_vector.py -q`

Expected: collection fails because `geodata.transform.vector` does not exist.

- [ ] **Step 3: Implement focused transform functions**

Move the eager behavior into `transform/vector.py`, but do not copy its private helper structure. Each function has three commented blocks: validate the native grid and values, call Rasterio, and wrap the result in GeoSave's native object.

For polygonization, use `nodata.fill_value(flags)` instead of reading attrs models directly. For rasterization, validate numeric values, call `nodata.check_fill_fits` for the fill and each unique property value, then cast once. Assert a regular `GeoBox` before using its CRS so static typing and runtime behavior agree.

Keep the methods' existing explicit typed signatures and delegate every argument by name. Do not replace those signatures with an untyped options dictionary.

- [ ] **Step 4: Verify GREEN, behavior, and typing**

Run: `uv run pytest tests/geodata/test_vector.py tests/geodata/transform/test_nodata.py -q`

Expected: all tests pass.

Run: `uv run basedpyright src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/transform/vector.py tests/geodata/test_vector.py`

Expected: `0 errors` in the new production surface; deliberate invalid-input tests use typed casts.

Run: `uv run mypy --ignore-missing-imports src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/transform/vector.py`

Expected: `Success: no issues found`.

- [ ] **Step 5: Record the checkpoint without committing dirty paths**

Record Task 3 as complete. Leave the new module untracked with the rest of the implementation because the receiving worktree intentionally carries the feature without implementation commits.

### Task 4: Integrated simplification and regression verification

**Files:**
- Modify: `tests/geodata/test_vector.py`
- Modify: `tests/geodata/test_core_smoke.py`
- Modify: `tests/geodata/test_raster_io_smoke.py`
- Verify all files named above.

**Interfaces:**
- Consumes: Tasks 1-3 as one public workflow.
- Produces: a readable, tested, type-clean GeoVector feature with the obsolete pipeline Manifest still removed.

- [ ] **Step 1: Update the end-to-end catalog test**

Use `path` as the explicit upsert key and remove every generated-identity assertion:

```python
updated = catalog.upsert(
    GeoVector.from_xarray(raster, path=asset, status="complete"),
    on="path",
)
assert len(updated.gdf.loc[updated.gdf.path.notna()]) == 1
```

Keep assertions for semantic geometry, native grid metadata, no Dask execution during registration, relocation-safe path resolution, and exact spatial matching.

- [ ] **Step 2: Verify the focused feature**

Run: `uv run pytest tests/geodata/test_vector.py tests/geodata/test_core_smoke.py tests/geodata/test_raster_io_smoke.py -q`

Expected: all focused tests pass.

- [ ] **Step 3: Verify code quality on the exact change boundary**

Run: `uv run ruff check src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/core/array.py src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/stack.py src/geosave_engine/geodata/transform/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py tests/geodata/test_vector.py tests/geodata/test_core_smoke.py tests/geodata/test_raster_io_smoke.py`

Expected: all checks pass.

Run: `uv run basedpyright src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/core/array.py src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/stack.py src/geosave_engine/geodata/transform/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py tests/geodata/test_vector.py`

Expected: `0 errors`.

Run: `uv run mypy --ignore-missing-imports src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/core/array.py src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/stack.py src/geosave_engine/geodata/transform/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py`

Expected: `Success: no issues found`.

- [ ] **Step 4: Verify geodata and repository boundaries**

Run: `uv run pytest tests/geodata -q`

Expected: the complete geodata suite passes.

Run: `uv run pytest -q`

Expected: record the actual full-suite result. The known unrelated collection failure for missing `geosave_engine.workflow.examples.reflectance` may remain; do not modify workflow code as part of GeoVector.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Record completion without an implementation commit**

Record exact test and type-check totals in the ledger. Do not stage or commit implementation paths because they overlap the user's pre-existing dirty worktree.
