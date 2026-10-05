# Reference Merger Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace live-cut raster assembly and positional training routing with
the implemented indexed reference, including full-parent validation.

**Architecture:** Retain Tiler's lazy cutting/windows. Replace the existing
TileMerger implementation with reference-driven numeric accumulation and native
parent metadata; migrate every active consumer in the same execution slice.

**Tech Stack:** NumPy, SciPy window functions, xarray, GeoPandas, Tiler for cuts,
PyTorch DataLoader, Lightning, existing raster/GeoParquet I/O.

**Spec:** [Prediction foundations](../specs/2026-10-04-prediction-foundations-design.md).
**Evidence:** [Backend comparison](../probes/2026-10-04-merger-comparison-report.md).
**Status:** Implemented and verified on 2026-10-05. Fresh review findings are
covered by regressions; the old live-cut assembly path is removed.

## Global constraints

- Keep native objects; geodata imports no torch and model imports no ml.
- Preserve `FramesSpec`/`TilesSpec`, lazy cutting, current padding modes, and
  exact spatial metadata/coordinates. Keep the Tiler dependency for cutting.
- IDs travel beside tensors. No new model-spec fields, output registry, aliases,
  random crop sampler, wheel build, publishing, commits, or workspace edits.
- Validation/test assemble logits; targets come directly from prepared parents.
- Preserve raw tile callback outputs and existing calibration weighting policy.
- Full-parent buffers are explicit, eager state required for this first backend.
  This does not promise bounded spatial chunking or unlimited scene size.

## Review focus

- Unknown/duplicate IDs, mismatched grids, and malformed validity masks must fail
  atomically before changing completion or sum buffers (Task 1).
- Valid zeros and invalid holes must remain distinguishable after input
  `nan_to_num`; masks are supplied before that conversion (Tasks 1 and 3).
- Completely missing parents and duplicate keys after completed-parent draining
  must remain detectable (Task 1).
- Frame identities and worker reopening must reproduce the same keys without
  retaining open raster handles (Task 2).
- Exact categorical labels, partial validation, and callback inputs must retain
  their semantics while ordinal target pairing is removed (Task 3).

Standalone weighting receives an explicit `overlap` keyword from the existing
recipe. This avoids guessing the overlap from a single persisted tile. SciPy
is already directly declared in the current `pyproject.toml`.

Supervised validity is a fourth native tensor beside inputs, targets, and IDs.
All consumers migrate together. Masks intersect finite named pixel inputs over
channels/time and remain explicit through evaluation transforms/zero filling.
Training ignores this unaugmented lineage mask; callback labels remain unchanged.

## Task 1: Replace the existing numeric merger

**Files:** `geodata/transform/tiling.py`,
`tests/geodata/transform/test_tiling.py`, `tests/geodata/io/test_geoparquet.py`.

**Interfaces:** Use the exact `TileMerger(reference, parents, window=None, overlap=0,
leading_dims=None)`, string-keyed `add(results, valid=None)`, completed-parent
`merge`, all-parent `pending`, and `finish` interfaces from the foundation spec.
Remain in the current module initially; avoid a forwarding alias. Remove
`Tiles.merger` and `_InFlight` when no active consumer needs them in Task 3.

- [x] Port current useful merger tests to reference/parent inputs, preserving
  multiple parents, non-square/padded edges, spatial coordinates, nodata/dtype,
  named leading dimensions, duplicate refusal, and atomic batch validation.
- [x] Add tests for explicit boolean masks, all-invalid contributions, partially
  invalid overlap, finite zero values, incomplete full jobs, and wholly missing
  parents. Recover grids after reference persistence/reorder without a live cut.
- [x] Run the new tests and observe failure against the old constructor/API.
- [x] Snapshot reference metadata at construction; validate IDs and parent
  coverage. Pixel windows and projection shapes must agree when present.
  Parent rasters supply target dimensions, coordinates, and real CRS.
- [x] Validate each entire batch before mutation. Keep seen IDs after draining.
  Require full-tile support, consistent leading shape and dtype per parent, and
  boolean spatial masks. Exclude nonfinite values and declared-invalid pixels
  from both sums and weights. Do not guess cropped/downsampled output grids.
- [x] Use float64 sum/weight buffers and emit the existing promoted result dtype
  `np.promote_types(incoming.dtype, np.float32)`; test float32/float64/int32
  behavior. Int32 results therefore become float64, permitting NaN holes.
  Integer categorical masks are not a dense prediction policy.
- [x] Build weighting with existing SciPy window functions, preserving all
  declared `StitchWindow` values and overlap-tile semantics. Record unavailable
  weighting cases explicitly rather than silently treating them as Hann.
- [x] Drain native DataArrays under parent IDs and restore exact target spatial
  coordinates. Generic output band meanings/units are assigned by consumers,
  never copied from input bands. Run the full tiling/I/O suites and scoped types.

## Task 2: Generate coherent keyed datasets

**Files:** `ml/datasets/tiles.py`, `ml/segmentation/supervised/data.py`,
their mirrored tests, `tests/model/encoder/test_context.py`.

**Interfaces:** `TileDataset(tiles, spec=None, *, parent_ids: Sequence[str])`
generates `.reference` and returns `(inputs, id: str)`. Supervised Dataset keeps
its manifest/spec constructor, exposes process-owned `.parents` and `.reference`,
and returns `(inputs, target, id: str, valid: Tensor)`.

- [x] Add failing tests for string-key collation, shuffled rows, overlapping
  frames, equal footprints, parent reorder, and reopened workers. Ordinary
  DataLoader collation returns a list of IDs beside batched tensors.
- [x] Preserve manifest `id` before flattening. No frames uses source ID;
  framed parents use `f"{source_id}/frame-{frame_index}"` within the snapshot.
  Validate generated IDs are unique; never parse IDs for time or geometry.
- [x] Build the cut and its reference coherently in the reading process. Do
  not accept a reordered table and silently pair `.iloc` with unrelated cut
  positions. Serialize metadata only; drop open parents/readers/cuts on worker
  handoff and regenerate identical identities when reopening.
- [x] Preserve lazy pixel reads and actual tile metadata used by location/time
  encoders. Training augmentation changes pixels/targets while IDs retain
  lineage, without claiming augmented output has the original tile grid.
- [x] Keep DataModule batch transforms/nan conversion unchanged while carrying
  explicit pre-conversion validity for evaluation. The supervised policy is
  derived from its named pixel inputs; test multiple groups and time axes.
  Do not expand public batch tuples without updating all consumer signatures.
- [x] Run dataset/data-module/context tests, including the existing slow worker
  tests where applicable. Pixel-only reference support does not silently grant
  arbitrary unaligned DataTree cutting: reject unresolved stack grids explicitly.

## Task 3: Migrate evaluation and delete obsolete routes

**Files:** `ml/segmentation/supervised/module.py`, `data.py`,
`ml/segmentation/callbacks.py`, `calibrate.py`, model README, architecture guide,
segmentation templates, all source-mirrored affected tests.

**Interfaces:** Dataset `.merger()` constructs the reference/parents merger;
`Module._mergers` keeps one logits merger per loader. Completed parent IDs retrieve
the target from `.parents[parent_id][target].dataset`. Keep segmentation forward
tensor-only and `predict_step` returning `(logits, ids)`.

- [x] Add failing tests for full-parent scoring once, unsorted completion,
  interrupted/sanity/limited validation, epoch resets, exact class IDs 2 and 4,
  and target nodata. Require target/output grids to agree before scoring.
- [x] Remove the target merger, averaging/rounding class IDs, ordinal-based
  logit/target pairing, and `.tolist()` key assumptions. Convert original parent
  labels using the same floating nodata/ignore-index rule as tile labels.
- [x] Preserve raw tile `logits`/`label` callback returns and current calibrator
  behavior. Do not score incomplete parents; full jobs explicitly call `finish`.
- [x] Update all current model/dataset/spec docstrings and generated examples.
  Remove `Tiles.merger`, old TileMerger routing/state, and obsolete representation
  tests. Preserve Tiler internals still earning their use for cutting and all
  unrelated native mosaic/concat/merge functions.
- [x] Search source/tests/guides/templates for `tiles.merger`, `TileMerger`,
  `index.tolist`, `_mergers`, and ordinal target pairing. Review each active hit;
  historical reports are not APIs. No compatibility alias or second legacy path.
- [x] Run affected tiling, I/O, model cuts, dataset, supervised, callback,
  encoder-context, and layering suites; scoped Ruff/BasedPyright; docs build;
  generated template config smoke; `git diff --check`.

Deliverable: one keyed reference-driven raster path for development inference
and full-parent evaluation. Task-specific detection/classification assembly and
production workflow APIs remain separate work.

## Verification

Final preservation suite: **259 passed, 12 skipped, 7 deselected**. This covers
reference/grid/persistence, string-keyed datasets, training/checkpoint reload,
Lightning prediction, full-parent validation/test, partial validation, callbacks,
calibration, encoder context, CLI/templates, and package layering. The explicit
forkserver smoke passed separately. Scoped Ruff and BasedPyright passed,
Zensical built, and `git diff --check` was clean.

Fresh review found periodic rather than established symmetric window weighting
and scalar input-coordinate collisions with predicted axes. Twelve declared
window cases with disagreeing predictions and two scalar band/time cases now
pass. No legacy merger route or target averaging remains in active consumers.

Breaking changes: required `TileDataset(..., parent_ids=...)`; string IDs beside
inputs; supervised `(inputs, target, ids, valid)` batches; standalone
`TileMerger(reference, parents, overlap=..., window=...)`. Full-parent eager
buffers and task-specific decoding/workflow scope remain explicit.

## Readability refinement, 2026-10-05

The user requested simpler fundamentals after migration. Cutting and reference
construction now live in `transform/tiling.py`; `TileMerger` and `StitchWindow`
live in `transform/stitching.py`. Consumers and mirrored tests move with ownership,
without compatibility exports.

The merger resolves signed windows into clipped source/destination slices once.
It validates a whole batch before accepting predictions, accumulates numeric sums
and coverage independently of xarray, then restores original coordinates on
drain. Remaining IDs per parent replace separate expected/seen collections.
Cut preparation is one explicit layout-and-padding operation; private docstrings
and stale commentary are shortened. Public behavior and the reference schema
remain the same.

Refinement checks: **467 passed, 12 skipped, 7 deselected**, covering all geodata
transforms and the earlier migration consumers. Scoped Ruff passed; BasedPyright
reported 0 errors, warnings or notes; Zensical built; `git diff --check` passed.
The standalone cutting/stitching smoke passed 98 cases before and after the
refactor. The prior explicit forkserver worker smoke remains valid for the
unchanged process-local dataset lifecycle.


## Dataclass and native-merger rewrite, 2026-10-05

The user rejected the helper-based implementation and asked why native Tiler
was not doing the numerical work. This supersedes the earlier internal numeric
accumulator and the readability refinement above.

`PixelWindow` owns the integer rectangle and its projection record. `_RasterCut`
owns one raster, its native Tiler layout and lazy padded data. `Tiles` is a
sequence dataclass. Native xarray slicing replaces generic map/rebuild operations,
while existing `map_groups` preserves stack metadata during padding. The union-
bounded generic helper that caused the IDE `Self` diagnostic is removed entirely.

`_ParentMerge` owns ID association, expected tiles and native `tiler.Merger` state.
Both cutters and mergers ask Tiler to calculate padding and recalculate the layout.
No spacing inference remains. `TileMerger` delegates numeric accumulation to native
Tiler; one added coverage channel allows normalization by valid contributions
instead of Tiler's default NaN-to-zero normalization. Model output channels are
flattened for native accumulation and restored before constructing native xarray
outputs. Public reference fields and forwarded prediction IDs remain unchanged.

Breaking change: `overlap` is now a required keyword for `TileMerger`, including
zero for disjoint tiles. Each parent's reference must match that regular recipe.
Native buffers cover the padded extent, with an additional coverage channel;
large-parent memory remains eager. There are no compatibility exports or custom
numeric accumulation engines.

Smoke evidence: 100 cutting/stitching tests passed, including two new metadata
regressions that failed on the previous implementation. Default native overlap
normalization returned 0 for the invalid-neighbour probe; native unnormalized sums
plus coverage returned 8, the correct contribution from the valid neighbour.
Final checks: **469 passed, 12 skipped, 7 deselected** in 79.18s, including
all geodata transforms, native-weight parity, persisted references, Lightning
prediction and full-parent validation/test. Ruff passed; BasedPyright reported
0 errors, warnings or notes; Zensical built; `git diff --check` passed.
