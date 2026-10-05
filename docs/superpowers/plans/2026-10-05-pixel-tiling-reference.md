# Native Pixel Tiling and Reference Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this
> plan task by task. Steps use checkbox syntax for tracking.

**Status:** Implemented following user approval on 2026-10-05; reviewed and verified; results recorded below.

**Goal:** Keep one indexed geospatial reference while delegating pixel tiling
and assembly to native Tiler/Merger.

**Architecture:** An existing GeoVector factory builds metadata from prepared
parents and native spatial layouts. Datasets use those layouts to read pixels;
training methods associate predictions with native tile IDs through the
reference. Remove the obsolete GeoSave pixel wrappers after migrating their
consumers together.

**Tech Stack:** tiler, xarray/Dask, GeoPandas/GeoParquet, odc-geo, Lightning.

**Spec:** [Native pixel tiling and a geospatial reference](../specs/2026-10-05-pixel-tiling-reference-design.md).

## Global Constraints

- No compatibility aliases or duplicate pixel execution paths.
- Geodata imports no torch; model does not import ml.
- Reference construction reads metadata only; dataset reads remain bounded.
- Keep Lightning tensor outputs, sample-ID tuples, and Kornia training crops.
- Keep original categorical targets and validity-aware full-parent metrics.
- No new prediction workflows, heads, dependencies, or wheel builds.
- Preserve unrelated changes in the shared checkout; do not stage or commit
  without a separate instruction.

## Review Focus

- Identical parent geometries and reordered rows must never exchange predictions.
- Plain rasters and WKT-only CRS must round-trip without invented georeferencing.
- Halo/fringe tiles must retain exact offsets and reconstruct the original extent.
- Named inputs with different band/time axes must share spatial cuts without
  forcing a two-dimensional native reader to return higher-dimensional arrays.
- Invalid overlapping predictions and incomplete epochs must not create valid
  values or misleading complete-scene metrics.

Each condition is assigned a behavioral check below.

## File ownership

| File | Responsibility after migration |
| --- | --- |
| `geodata/core/vector.py` | Metadata-only `GeoVector.from_tiles` factory |
| `model/spec/cuts.py` | Recipe creates native spatial Tiler from a shape |
| `ml/datasets/tiles.py` | Named model inputs and persisted IDs from spatial windows |
| `ml/segmentation/supervised/data.py` | Prepared parents, layouts, padding, and lazy reopening |
| `ml/segmentation/supervised/module.py` | Task-owned validity, native assembly, full-scene metrics |
| `geodata/transform/tiling.py`, `stitching.py` | Removed when consumers are migrated |

### Task 1: Produce the metadata reference independently

**Files:** Modify `src/geosave_engine/geodata/core/vector.py`; test
`tests/geodata/core/test_vector.py` and
`tests/geodata/io/test_geoparquet.py`.

**Consumes:** Original parents keyed by scene/frame ID, matching native 2D
Tiler layouts, optional per-parent before/after padding widths.

**Produces:** `GeoVector.from_tiles(parents, layouts, *, padding=None)` returning
one native GeoDataFrame with the columns and null policy specified in the spec.

- [x] Add failing tests for sample ID uniqueness, native `tile_id`, parent
  reordering, and two different parents with identical geometry. Assert zero
  Dask computation during metadata construction.
- [x] Add exact-grid tests for rotated transforms, WKT-only CRS, halo offsets,
  requested support, mixed referenced/plain parents, and an all-plain table.
- [x] Add a GeoParquet round trip using `index=False`; restore the lookup with
  `set_index("id", drop=False, verify_integrity=True)` and shuffle rows. Assert
  that IDs, grid fields, and absent CRS survive.
- [x] Run the new tests and confirm the missing factory is the failure.
- [x] Implement the factory using native `get_tile_bbox`, original-parent grid
  translation, and existing projection conventions. Use existing classmethod
  style and concise public docstrings; introduce no pixel wrapper.
- [x] Run `uv run pytest tests/geodata/core/test_vector.py
  tests/geodata/io/test_geoparquet.py` and scoped Ruff/type checks.

**Deliverable:** Independently usable and persisted tile metadata, without
changing current dataset or merger behavior.

### Task 2: Migrate pixel consumers and remove obsolete wrappers

**Files:** Modify `src/geosave_engine/model/spec/cuts.py`,
`src/geosave_engine/ml/datasets/tiles.py`,
`src/geosave_engine/ml/segmentation/supervised/data.py`,
`src/geosave_engine/ml/segmentation/supervised/module.py`, and
`src/geosave_engine/geodata/transform/__init__.py`.
Remove `src/geosave_engine/geodata/transform/tiling.py` and `stitching.py`.
Update their mirrored tests, `tests/model/spec/test_cuts.py`,
`tests/ml/datasets/test_tiles.py`, and
`tests/ml/segmentation/supervised/test_data.py` / `test_module.py`.
Update live examples in `docs/guides/architecture.md`,
`docs/guides/workflows.md`, and relevant templates only where affected.

**Consumes:** Task 1's table and native Tiler/Merger.

**Produces:** `TilesSpec.layout(shape: tuple[int, int]) -> Tiler`; per-parent
native layouts used by existing datasets and dense evaluation; existing
prediction tuples with unchanged model-facing tensors.

- [x] Inspect all current wrapper imports with
  `rg 'TileMerger|PixelWindow|transform\.(tiling|stitching)|tiles\.cut' src tests docs`.
  Separate historical design notes from executable consumers. Read current
  evaluation tests before changing behavior.
- [x] Rewrite/add a small-model end-to-end test using shuffled IDs and two
  identically georeferenced parents. Tile inference plus native assembly must
  match full-parent inference. Cover boxcar and Hann with explicit halo setup.
- [x] Pin reader behavior with named Dataset/DataTree inputs, differing leading
  band/time axes, bounded Dask reads, and DataLoader worker reopening. Confirm
  input ordering and existing model context survive.
- [x] Pin evaluation policy: original categorical labels, invalid overlap
  exclusion, uncovered nodata, and incomplete-parent handling. Keep detection
  and classification output association tests without adding raster merging
  to those tasks.
- [x] Run the focused tests against the proposed interfaces and identify
  migration failures before implementation.
- [x] Replace `TilesSpec.cut` with the native shape factory. Keep temporal
  `FramesSpec.cut` separate. Move recipe literal types out of obsolete wrappers.
- [x] Make prepared datasets retain parents, layouts, and native padding.
  Read each input's spatial window without materializing the parent. Build
  the reference once; reopen workers from the same shapes/recipe and snapshot.
- [x] In the existing supervised method, resolve sample IDs to `(parent_id,
  tile_id)` and use native mergers. Keep validity, completion, interpretation,
  and raster metadata restoration local to their existing owners. Do not add
  a generic replacement merger or spread this behavior across speculative
  modules.
- [x] Remove both obsolete pixel modules and their exports/imports after
  migrating consumers. Replace wrapper-specific tests with behavioral tests;
  retain round-trip, lazy, and small-model coverage.
- [x] Update API examples and affected template calls. Leave historical specs
  intact and point their supersession status to the accepted new design.
- [x] Run focused tests:
  `uv run pytest tests/model/spec/test_cuts.py tests/ml/datasets
  tests/ml/segmentation/supervised tests/geodata/core/test_vector.py
  tests/geodata/io/test_geoparquet.py`.
- [x] Run scoped Ruff, configured BasedPyright, `git diff --check`, and a docs
  build. Repeat the stale-import search. Broaden tests only if changed contracts
  or failures justify it; do not build wheels.

**Deliverable:** One native pixel path with the persistent reference contract
and existing training/evaluation behavior preserved.

## Exit criteria

The table alone recovers each tile's geospatial location. Native layouts alone
determine pixel placement. Shuffling cannot change associations; padding and
validity tests pass; obsolete wrappers have no live consumers. Report actual
checks and any remaining limitations before claiming the migration complete.


## Implementation record

- `GeoVector.from_tiles` now produces the metadata-only reference. Datasets
  retain `id` as a column and use it as their lookup index.
- `TilesSpec.layout` returns native Tiler; explicit overlapping halos stay in
  dataset setup. Constant-padding NaN semantics are preserved.
- Named-input readers use native channel-aware Tiler callbacks for bounded
  reads and fringe padding, restoring each variable's leading axes afterward.
- Supervised evaluation uses native Merger with a coverage channel. Completion,
  validity, original categorical labels, and scoring remain method-owned.
- Removed obsolete tiling/stitching wrappers and migrated live docs, templates,
  encoder context tests, and callback tests. Wrapper-specific defensive API
  tests were replaced by reference and native reader/evaluation behavior tests.
- Final combined regression: 453 passed, 7 deselected. Explicit slow DataLoader worker
  check: 1 passed. Ruff and documentation build passed.
- Full repository BasedPyright reported unrelated diagnostics across untouched
  code and existing negative/type-stub tests; changed library files are checked
  separately. No unrelated type cleanup is included.

- Final review identified and resolved native constant halo fill mismatches and
  unsafe one-pixel taper overlap. Regression tests were observed failing, then
  44 native reader/recipe tests passed. Reviewer confirmed both fixes.
- Scoped BasedPyright for all five changed library modules: 0 errors,
  0 warnings, 0 notes.
