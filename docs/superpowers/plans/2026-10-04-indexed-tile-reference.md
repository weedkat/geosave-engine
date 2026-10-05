# Indexed Tile Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose an indexed native reference for geographic and pixel-only tiles,
then use it to compare merger approaches before migrating training or prediction.

**Architecture:** `Tiles` generates placement records without reading pixels.
Each prediction carries the ID of its reference row. Pixel-window placement is
common to both raster kinds; geographic metadata is present only when real.

**Tech Stack:** xarray, GeoPandas/pandas, odc-geo, PyTorch/Lightning, existing
GeoParquet I/O, and the current tiler dependency.

**Spec:** `docs/superpowers/specs/2026-10-04-indexed-tile-reference-design.md`

**Status:** Task 1 implemented. Task 2's comparison is complete; the
recommended Tiler-window/reference-accumulator backend migrated on 2026-10-05.
The [prediction foundations plan](2026-10-04-prediction-foundations.md)
coordinates the sequence. Its [migration plan](2026-10-04-reference-merger.md)
reads categorical targets directly from prepared parents rather than averaging
them through a numeric target merger.

The user explicitly requires stale-code cleanup for the pivot. Task 1 is not
the end state. The migration inventory and completion gate below are required
inputs to the follow-up assembler plan.

## Global Constraints

- Keep native pandas/GeoPandas, xarray, and Lightning objects. No new record,
  prediction, or training wrapper is required.
- Rasters without georeferencing remain supported. Their CRS, transform, and
  geographic geometry must not be fabricated.
- Keep `FramesSpec`, `TilesSpec`, and current model-spec tiling fields.
- Metadata construction must not compute source pixels.
- Preserve the existing `Tiles` / `TileMerger` and training behavior in Task 1.
- Use existing dependencies; do not build local wheels or publish artifacts.
- Preserve unrelated working-tree changes. Commit only if explicitly requested.

## Review Focus

- A mixed reference has geographic and null geometries; null rows must not
  acquire the frame's geographic CRS as their pixel-grid CRS.
- Code-only and WKT-only grid CRS metadata must survive Parquet null/array
  normalization without ambiguous truth-value checks.
- Source order and table order can change; persistent keys must still identify
  the same parent-local tiles.
- Padding produces negative offsets; the recorded window must describe the
  full tile before later target clipping.
- Lazy source arrays must remain unevaluated during reference construction.

## Files and responsibilities

| File | Change |
| --- | --- |
| `src/geosave_engine/geodata/transform/tiling.py` | Add `Tiles.reference`; expose current layout metadata |
| `tests/geodata/transform/test_tiling.py` | Reference identities, pixel windows, geographic grids, laziness |
| `tests/geodata/io/test_geoparquet.py` | Reference persistence with projection arrays and nullable geometry |
| `src/geosave_engine/model/spec/cuts.py` | Add a concise reference example to `TilesSpec`; preserve its interface |
| `src/geosave_engine/ml/datasets/tiles.py` | Document the proposed keyed sample association; do not migrate behavior |
| `docs/guides/architecture.md` | Explain identity, pixel placement, and optional world georeferencing |
| This plan and its spec | Record the backend comparison and remaining decisions |

## Task 1: Expose and persist the candidate reference

**Interface:**

```python
def reference(self, parent_ids: Sequence[str]) -> gpd.GeoDataFrame: ...
```

Produces rows in current tile order with `id`, `parent_id`, signed `row_off`
and `col_off`, `height`, `width`, `geometry`, and nullable existing projection
fields. IDs use `f"{parent_id}/tile-{local_tile_id}"` as specified; downstream
code looks them up rather than parsing them. No constructor or dataset changes.

- [x] Add failing reference tests in `tests/geodata/transform/test_tiling.py`:
  `test_reference_identifies_each_parent_tile`,
  `test_reference_preserves_parent_ids_when_sources_reorder`,
  `test_reference_records_padding_offsets`,
  `test_reference_reconstructs_exact_geographic_grids`,
  `test_reference_describes_pixel_only_tiles_without_georeferencing`, and
  `test_reference_preserves_null_geometry_in_mixed_rasters`.
  Assert row count equals `len(tiles)`, IDs are unique, parent identities match
  `locate`, and both the derived world grid and pixel window match actual tiles.
- [x] Add `test_reference_validates_parent_ids`: reject count mismatch, repeated
  or empty parent IDs before constructing rows.
- [x] Add `test_reference_does_not_compute_pixels` using a Dask-backed raster
  whose delayed pixel read raises if executed; reference construction succeeds.
- [x] Run `uv run --no-sync pytest tests/geodata/transform/test_tiling.py -q`;
  new tests must fail for the missing method while existing tests pass.
- [x] Implement the method in `geodata/transform/tiling.py`. Reuse `locate`,
  `get_tile_bbox`, existing before-padding, and each parent's geobox metadata.
  Normalize geometry to EPSG:4326 only when georeferencing exists. Return no CRS
  for an all-unreferenced table. Do not read private tiler state outside the
  current wrapper or introduce a second window-generation algorithm.
- [x] Add persistence tests in `tests/geodata/io/test_geoparquet.py` using
  existing `io.geoparquet.write/read`: all-geographic, all-pixel-only, and mixed
  records. Preserve `id` as a column, write `index=False`, shuffle restored rows,
  and perform lookup with `set_index('id', verify_integrity=True)`.
- [x] Add a WKT-only non-EPSG grid fixture. Assert exact shape/affine/CRS recovery
  after Parquet array/null normalization and no geographic metadata on plain rows.
- [x] Add a keyed batch smoke test in `tests/ml/datasets/test_tiles.py` using
  ordinary `DataLoader` collation, shuffled rows, and identical footprints in
  separate parents. The probe constructs samples beside the existing dataset;
  it does not change the production dataset's integer routing contract.
- [x] Update the `TilesSpec` example and architecture guide with the public
  reference object, pixel-only handling, and the distinction between permanent
  IDs and DataLoader positions. Mark keyed training migration as future work.
- [x] Run focused suites:
  `uv run --no-sync pytest tests/geodata/transform/test_tiling.py tests/geodata/io/test_geoparquet.py tests/ml/datasets/test_tiles.py tests/model/spec/test_cuts.py -q`.
  Require all tests to pass; existing tiling and merging behavior must remain.
- [x] Run `uv run --no-sync ruff check src/geosave_engine/geodata/transform/tiling.py src/geosave_engine/model/spec/cuts.py src/geosave_engine/ml/datasets/tiles.py tests/geodata/transform/test_tiling.py tests/geodata/io/test_geoparquet.py tests/ml/datasets/test_tiles.py`,
  `uv run --no-sync basedpyright src/geosave_engine/geodata/transform/tiling.py`,
  and `git diff --check`. Require no new lint/type errors or whitespace errors.
  Review only the task's changes against the shared-tree baseline; do not stage
  unrelated modifications.

Deliverable: a usable reference API with geographic and pixel-only round trips.
This is independently useful even if the existing merger remains unchanged.

## Task 2: Compare merger candidates before committing to a migration

**Consumes:** Task 1's reference, current `Tiles.merger`, existing raster I/O.

**Produces:** Evidence and a reviewed assembler contract in the spec. This task
does not prescribe a new public class, `blend` signature, or prediction flow.

- [x] Assemble one small georeferenced raster and one pixel-only raster using
  a tiny Torch model and shuffled persistent tile IDs. Use direct full-scene
  pointwise prediction and the current merger as the placement baselines.
- [x] Compare the three spec candidates in throwaway code: current tiler
  routing/merging, tiler-generated windows with reference-driven accumulation,
  and native window generation with reference-driven accumulation. Show the
  caller and maintainer code for each candidate.
- [x] Verify non-square dimensions, trailing and leading padding, zero overlap,
  averaging, Hann weighting, incomplete inputs, and duplicate contributions.
  Preserve null georeferencing in pixel-only outputs and actual grids otherwise.
- [x] Add an invalid-pixel probe: holes contribute no values or weights; valid
  zero logits remain covered. Demonstrate where each candidate needs additional
  code rather than claiming that default averaging handles validity.
- [x] Compare in-memory and persisted raw-output assembly for referenced data
  using existing I/O. Record what additional parent metadata is needed to merge
  after the live `Tiles` is discarded. Investigate a native persistence format
  for pixel-only outputs before proposing a mandatory asset writer.
- [x] Record elapsed time and allocated accumulator bytes for increasing parent
  shapes with the same output channels and tile recipe. Identify full-parent
  buffers and whether a candidate can genuinely process bounded spatial chunks.
- [x] Record which candidates preserve current edge and weighting semantics,
  which support validity directly, and the code needed to adapt each.
- [ ] Update the spec with the selected contract only after the comparison is
  reviewed. Create a separate implementation plan for assembler integration,
  keyed validation/test, decoding, and optional persistence. Do not migrate them
  as undocumented follow-up work inside Task 1.

## Required follow-up: atomic consumer migration and stale-code removal

This is the scope the next implementation plan must cover once Task 2 resolves
the assembler contract. Concrete replacement signatures belong in that plan;
do not prescribe them before the comparison.

| Files | Migration / cleanup |
| --- | --- |
| `ml/datasets/tiles.py`, `ml/datasets/__init__.py` | Return reference keys; update input/return annotations and examples; remove stale positional routing imports/helpers |
| `ml/segmentation/supervised/data.py` | Preserve manifest/frame identities before flattening; generate matching reference keys in workers; replace live-cut merger access; update batch types |
| `ml/segmentation/supervised/module.py` | Replace `.tolist()` assumptions and ordinal-based logit/target pairing; construct assembly from the owning reference; preserve full-parent metrics |
| `geodata/transform/tiling.py` | Remove obsolete `TileMerger` implementation/routing after consumers migrate; retain cutting internals and edge behavior still used by the selected backend |
| `ml/segmentation/callbacks.py`, `ml/segmentation/calibrate.py` | Verify existing step-output contracts; update only genuine ID/assembly dependencies, without changing threshold ownership implicitly |
| `model/README.md`, `model/spec/cuts.py`, source docstrings | Rewrite reconstruction examples and explanations when the replacement actually exists |
| `templates/workspaces/segmentation/`, architecture/workspace guides | Keep generated examples consistent with the selected training and prediction contract |
| `tests/geodata/transform/test_tiling.py`, `tests/ml/datasets/test_tiles.py` | Port useful behavior tests; remove tests enforcing obsolete positional representation only |
| `tests/ml/segmentation/supervised/test_data.py`, `test_module.py` | Reference identity through frames/workers; full-scene scoring; stable logit/target association |
| `tests/model/encoder/test_context.py`, callback tests | Preserve time/location tensor behavior and logging/calibration outputs |
| `pyproject.toml`, `uv.lock` | Remove `tiler` only when the chosen implementation has no remaining use |

- [x] Capture a task-local baseline for the shared working tree before editing
  source. Do not mistake unrelated pending refactors for work to delete.
- [x] Make source and temporal-frame identity explicit before `Dataset._cut`
  flattens parents. Test reopened worker datasets reproduce the same reference
  keys and preserve process-local raster ownership.
- [x] Test repeated tile keys across distinct loaders/splits remain scoped to
  their owning references. Test equal footprints at different output frames are
  assembled separately.
- [x] Add a real raster + small Torch model integration test: shuffle keys,
  persist/reload the reference, discard the original cut, and reconstruct using
  only the reference and explicit parent metadata.
- [x] Preserve completed-parent validation/test metrics and interrupted-epoch
  behavior. Read labels from the original prepared parent, test exact class
  values and invalid pixels, and remove target averaging/rounding. Require
  target/output grid agreement; do not resolve label conflicts by blending.
- [x] Test completion against all expected reference IDs, including a wholly
  missing parent. Preserve exact original coordinate labels and named leading
  axes, in addition to the output affine grid and dtype.
- [x] Retain raw tile `logits`/`label` outputs consumed by current callbacks
  unless a separately reviewed calibration/logging change deliberately replaces
  that behavior. Training augmentation preserves identity without claiming the
  augmented image occupies the original geographic grid.
- [x] Search current source, mirrored tests, guides, model README, templates,
  and dependency files for `TileMerger`, `tiles.merger`, `dataset.merger`,
  `dataset.tiles.merger`, `index.tolist`, ordinal-based pairing, and removed
  import paths. Classify each remaining hit; history is not a live API.
- [x] Remove superseded execution paths and compatibility aliases. Internal
  `Tiles.locate` can remain if cutting still requires it; deletion is based on
  remaining callers, not the name alone.
- [x] Run the source-mirrored tiling, general dataset, supervised training,
  encoder context, callback, and spec suites; scoped lint/type checks; docs
  build; and a fresh generated-workspace configuration smoke. No wheel build.

Completion means all active consumers use the chosen reference contract and
obsolete routes have been removed. It does not mean every occurrence of the
word `tile` or the existing `Tiles` name must disappear.

## Implementation verification

Reference regression tests cover parent-local identity, signed padding,
non-square and rotated grids, laziness, WKT-only CRS, absent CRS, and
geographic/pixel-only/mixed GeoParquet round trips. Shuffled DataLoader probes
recover matching classification and detection context from persisted IDs.

The [comparison report](../probes/2026-10-04-merger-comparison-report.md) records
all three backend candidates, validity/completion tests, persisted assembly,
and time/buffer measurements. The [migration plan](2026-10-04-reference-merger.md)
now migrates the production datasets and merger with original-parent scoring.
The combined preservation run passed **337 tests, 9 deselected**.

## Additional discovery: native outputs and workflow scope

The [full-flow report](../probes/2026-10-04-predict-report.md) records a completed
local Prefect flow over 36 tiles using the implemented public reference API,
native dense/classification/detection outputs, and geographic/pixel-only
persistence. Averaging and Hann matched direct pointwise prediction.

The earlier tensor-only terminal limitation is resolved: declared native
containers and mixed terminal outputs now remain whole. Task-specific decoding
and production assembly still need their own consumers. Completed jobs by output
family and reusable inference are candidates; this change introduces no generic
dispatch registry or production prediction flow.

The consumer migration completed on 2026-10-05: 259 preservation tests passed,
with a separately verified forkserver worker smoke. See the reference merger
plan for API changes and review regression evidence.
