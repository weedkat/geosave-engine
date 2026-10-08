# Items From Saved Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build STAC Items from the files a writer saved, and carry Collections in the table's file metadata.

**Architecture:** `stac/item.py` opens each saved file once with `read_raster`; that raster yields the asset, footprint, time and id. `stac/asset.py` builds one Asset from a raster and opens nothing. `stac/table.py` writes Collections through stac-geoparquet and computes their extent from the rows.

**Tech Stack:** xarray, PySTAC 1.14.3, stac-geoparquet 0.8.2, existing `read_raster` / `read_stack`.

**Spec:** `docs/superpowers/specs/2026-10-08-items-from-saved-files-design.md`

## Global Constraints

- Public names: `create_item`, `create_items`, `create_stack_items`, `create_collection`, exported from `geodata.stac`.
- `collection=` takes a `pystac.Collection`, never a string. PySTAC's `set_collection` / `add_items` are not used.
- A file is opened only in `item.py`. No pixel is computed while describing.
- No compatibility aliases: `gs.to_items`, `item.from_raster`, `from_files`, `from_stack` are deleted.
- No commits: the working tree carries an unrelated refactor.
- Baseline: `uv run pytest` gives 1744 passed.

## Review Focus

1. A writer-level `encoding` override is described as the file stores it. (Task 2)
2. Two files of one instant claiming the same asset key raise instead of one silently replacing the other. (Task 2)
3. A timeless group of a stack takes the dated groups' span; a stack with no dated group raises. (Task 3)
4. A row naming a Collection that was not passed raises before anything is written. (Task 4)
5. Part files giving one Collection id different descriptions raise; different extents union. (Task 4)
6. Selected assets stating different open options raise in `table.load`. (Task 5)

---

### Task 1: `asset.py` builds one Asset and nothing else

**Files:** Modify `src/geosave_engine/geodata/stac/asset.py`, `src/geosave_engine/geodata/stac/band.py`; Test `tests/geodata/stac/test_asset.py`, `tests/geodata/stac/test_band.py`.

**Interfaces:** Produces `asset.from_raster(raster: xr.Dataset, href: str | PathLike[str]) -> pystac.Asset` (no time fields), `asset.default_key(raster: xr.Dataset) -> str`.

- [x] Rewrite the asset tests to drop `start_datetime` / `end_datetime` and the built-versus-file agreement test; drop the `decode_cf` band test.
- [x] Inline `_describe` into `from_raster`, remove the time block, the Asset branch of `default_key` and the `.encoding` merge in `band.from_variable`.
- [x] `uv run pytest tests/geodata/stac/test_asset.py tests/geodata/stac/test_band.py -q`

### Task 2: `create_collection`, `create_item`, `create_items`

**Files:** Rewrite `src/geosave_engine/geodata/stac/item.py`; Modify `src/geosave_engine/geodata/stac/__init__.py`; Test `tests/geodata/stac/test_item.py`.

**Interfaces:**
- Consumes: `asset.from_raster`, `asset.default_key`, `extensions.schemas`, `read_raster`.
- Produces: the three signatures in the spec, `STACK_ID`, `ItemTime`.

- [x] Failing tests: store is one Item; one Item per instant; split bands are one asset per variable; id template and its repeat error; timeless needs `datetime`; `collection=forest` sets the id and writes no link; `create_item` with caller keys; duplicate asset key raises; an `encoding` override reads as stored; no pixels computed.
- [x] Implement with one private record:

```python
class _Saved(NamedTuple):
    """One saved raster, opened lazily."""

    href: str                    # '/data/forest/forest_20250601T103031/red.tif'
    raster: xr.Dataset           # what read_raster hands back for it
    options: dict[str, Any]      # options a reader needs to open it again, as {'group': 'optical'}
```

- [x] `uv run pytest tests/geodata/stac/test_item.py -q`

### Task 3: `create_stack_items`

**Files:** Modify `src/geosave_engine/geodata/stac/item.py`; Test `tests/geodata/stac/test_item.py`.

- [x] Failing tests: COG mapping gives each group's Items with `collection` = group, ids prefixed, `geosave:stack`; a timeless label takes the optical span; a grouped store gives one Item per group with `xarray:open_kwargs.group`; no dated group raises.
- [x] Implement over the same `_Saved` records; a store's groups come from `read_stack`.
- [x] `uv run pytest tests/geodata/stac/test_item.py -q`

### Task 4: Collections in the table

**Files:** Modify `src/geosave_engine/geodata/stac/table.py`; Test `tests/geodata/stac/test_table.py`.

**Interfaces:** Produces `write(..., collections: Sequence[pystac.Collection] | None = None)`, `read_collections(path, *, storage_options=None) -> dict[str, pystac.Collection]`.

- [x] Failing tests: two Collections round trip with license and providers; the stored extent is the rows' bounds and time range while the passed object keeps its open extent; a dangling id raises; two part files union extents; conflicting descriptions raise.
- [x] Implement; `uv run pytest tests/geodata/stac/test_table.py -q`

### Task 5: `table.load` selects, then opens

**Files:** Modify `src/geosave_engine/geodata/stac/table.py`; Test `tests/geodata/stac/test_table.py`.

- [x] Failing test: rows of two groups of one store raise `ValueError` naming both option sets.
- [x] Split `_data_assets(rows, wanted) -> list[DataAsset]` out of `load`.
- [x] `uv run pytest tests/geodata/stac/test_table.py -q`

### Task 6: Remove the object-based entry points and migrate callers

**Files:** Modify `src/geosave_engine/geodata/core/{raster,array,stack}.py`, `src/geosave_engine/workflow/tasks/labels.py`, `docs/guides/{architecture,workflows}.md`; Tests under `tests/geodata/core`, `tests/geodata/io/test_remote.py`, `tests/geodata/stac/test_dataflow.py`, `tests/ml/segmentation/supervised/test_data.py`, `tests/cli/core/test_workspace.py`.

- [x] Delete `to_items` from the three accessors and their `ItemTime` imports; move their tests to `create_items` / `create_stack_items`.
- [x] `labels.py` calls `create_item({"label": path}, id=sample_id)`.
- [x] `uv run pytest`, `uv run pytest -m integration tests/geodata/stac`, `uv run ruff check src tests`, `scripts/check_docstrings.py` and BasedPyright on the changed modules.
