# Prediction Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support native mixed model results and persisted tile identity, then
select and migrate raster assembly using measured behavior.

**Architecture:** Terminal declarations determine model results; IDs beside
tensors identify native reference rows. Raster assembly consumes the reference
and native parent metadata, independently of model execution or task decoding.

**Tech Stack:** PyTorch, xarray, GeoPandas, Dask, existing GeoParquet I/O,
Lightning, Prefect, and the current tiler dependency until selection.

**Spec:** [Prediction foundations](../specs/2026-10-04-prediction-foundations-design.md)
and [indexed reference details](../specs/2026-10-04-indexed-tile-reference-design.md).

**Status:** Tasks 1 and 2 implemented. Task 3's comparison is complete and
recommends retaining Tiler windows with reference-driven accumulation. Its
[backend-specific migration plan](2026-10-04-reference-merger.md) was authorized
and verified on 2026-10-05; Task 4 is complete. Task 5 remains subsequent workflow discovery.

## Global constraints

- Keep native pandas/GeoPandas, xarray, PyTorch, and Lightning objects.
- IDs travel beside model inputs and outputs; never pass them into model calls.
- Rasters without georeferencing remain supported. Their CRS, transform, and
  geographic geometry must not be fabricated.
- Keep `FramesSpec`, `TilesSpec`, and current model-spec tiling fields.
- Metadata construction must not compute source pixels.
- Keep supervised segmentation `forward()` tensor-only and retain `predict_step`.
- No new output registry, wrapper, task-selector YAML, or generic training loop.
- Preserve unrelated changes, generated `workspace/`, and release workflows.
- No wheel build, publishing, automatic staging, or commits.

## Review focus

- Terminal dict/list/tuple results must remain whole native objects; mixed
  terminals must not lose results or leak intermediate features (Task 1).
- Null projection cells from Parquet must not fabricate CRS, and source order
  changes must not change a parent's own tile keys (Task 2).
- Negative padding offsets, valid zero values, invalid holes, and completely
  missing parents must have explicit, tested assembly behavior (Task 3).
- Frame identity and worker reopening must reproduce keys; augmented training
  IDs track lineage without claiming the transformed image's geometry (Task 4).
- Categorical targets must never be numerically blended; empty detection tiles
  must remain distinguishable from unprocessed tiles (Tasks 4 and 5).

## Task 1: Native terminal model results

**Files:**

- Modify: `src/geosave_engine/model/chain/step.py`, `chain.py`.
- Modify: `src/geosave_engine/model/README.md`.
- Test: `tests/model/chain/test_step.py`, `test_chain.py`.
- Preservation checks: `tests/model/head/`, `tests/model/release/test_artifact.py`,
  `tests/ml/segmentation/supervised/test_module.py`.

**Interfaces:**

- Consumes: current `chain_step(head=True)` and `Step.invoke(module, context)`.
- Produces: `Step.result_type: type | None`, appended after existing fields,
  holding the terminal annotation; `None` for named-output steps.
- `Step.invoke(...) -> object`; validates the annotated outer terminal type.
- `ModelChain.forward(self, *args: object, **kwargs: object) -> object`;
  one terminal returns unchanged, multiple terminals return
  `dict[str, object]`, no terminals return named produced values.
- `head=True` remains incompatible with explicit `outputs`.

- [x] Capture a scoped pre-edit baseline of these files, including untracked
  files used by tests. Never use Git HEAD as the whole shared-tree baseline.
- [x] Add `test_terminal_native_list`, `test_terminal_native_dict`,
  `test_terminal_tuple_is_not_unpacked`, and `test_terminal_outer_type_mismatch`.
  Native list contains two detection dictionaries; empty list/dictionaries are
  accepted; incorrect Tensor-for-list return is refused at invocation.
- [x] Replace the obsolete tensor-only annotation rejection test with
  `test_terminal_requires_runtime_return_annotation`. Reject missing/None/Any
  annotations and annotations without a usable runtime outer type. Keep named
  output arity and self-cycle tests unchanged.
- [x] Add chain tests `test_single_native_terminal_is_returned_unchanged`,
  `test_mixed_terminals_keep_stage_names`, and
  `test_mixed_terminals_preserve_gradients`. Assert keys are exactly `cover` and
  `objects`; the encoder's `features` key is absent; backward on cover reaches
  the encoder. Keep single/two tensor and no-head behavior tests unchanged.
- [x] Run `uv run --no-sync pytest tests/model/chain -q`; new tests must fail on
  the current tensor-only decorator or tensor-based result dispatch.
- [x] Store the terminal annotation on `Step`, validate through the existing
  outer-type helper, and use `step.method.head` in chain dispatch. Terminal
  tuples bypass named-output unpacking. Do not alter dependency selection.
- [x] Update `ModelChain.__repr__` and concise public docstrings to describe the
  actual terminal type. Add `test_repr_reports_native_terminal_type`.
- [x] Update model README with bare detection and mixed output examples. Explain
  that named nonterminal detector values are intermediate when terminals exist.
- [x] Run `uv run --no-sync pytest tests/model/chain tests/model/head tests/model/release/test_artifact.py tests/ml/segmentation/supervised/test_module.py -q`.
  All tests pass; existing tensor model save/load and training remain valid.
- [x] Run scoped Ruff, BasedPyright on `model/chain`, and `git diff --check`.
  Review actual annotation/runtime behavior against the spec; broad `object`
  annotations must not become a reason to drop runtime checking.

Deliverable: native mixed terminal composition, with no changes to geometry,
training methods, release schema, or production workflows.

## Task 2: Persist the indexed reference

**Files / exact interfaces / tests:** Execute Task 1 of the existing
[indexed reference plan](2026-10-04-indexed-tile-reference.md).

Produces `Tiles.reference(self, parent_ids: Sequence[str]) -> gpd.GeoDataFrame`.
Its task already pins row fields, generated IDs, signed offsets, geographic and
pixel-only grids, laziness, WKT-only CRS, and GeoParquet round trips.

- [x] Execute the reference task's red/green test cycle and focused checks.
- [x] Preserve production dataset integer keys and live merger for this slice.
  Its keyed DataLoader experiment is a probe, not a partial dataset migration.
- [x] Confirm `height,width` describe the cut's input tile support. Current
  exact-grid outputs reuse it; outputs with another grid must not claim it.
- [x] Confirm lookup after persistence/shuffle uses the `id` column and validates
  uniqueness. An all-pixel-only table uses existing I/O directly without `.gs`.

Deliverable: metadata-only native reference API. This task is independently
useful and does not depend on Task 1's changed result type.

## Task 3: Select the reference-driven raster backend

**Files:**

- Create: `docs/superpowers/probes/2026-10-04-merger-comparison.py`.
- Create: `docs/superpowers/probes/2026-10-04-merger-comparison-report.md`.
- Update: reference spec and this plan with the measured decision.

**Consumes:** Task 2 reference, native parent metadata, existing `Tiles.merger`,
and the target `TileMerger(reference, parents, ...)` contract in the foundation
spec. Comparisons remain disposable until the decision is reviewed.

- [x] Implement callable `compare() -> dict[str, object]` in the probe. Run the
  same inputs through existing Tiler/Merger, Tiler windows plus reference-driven
  accumulation, and native windows plus reference-driven accumulation. Show
  caller and maintainer code for each in the report.
- [x] Use georeferenced and pixel-only 12x16 parents, 8x8 tiles, overlap 0 and 4,
  and a non-square 6x8 recipe. Test signed padding, shuffled IDs, same geometry
  in different parents, exact coordinate restoration, and named leading axes.
- [x] Compare mean and Hann to direct pointwise full-parent inference at
  absolute tolerance 1e-6 on valid pixels. A valid zero value has positive
  weights; an invalid hole remains nodata. Record each candidate's actual
  extra code for validity rather than giving unsupported behavior a pass.
- [x] Reject duplicate/unknown IDs and mismatched output shape atomically before
  mutating accumulators. Repeated keys after completion also fail. `pending`
  includes a wholly missing parent and `finish()` refuses unfinished jobs.
- [x] Discard live cuts, reload reference and raw outputs in different orders,
  reopen explicit parent assets, and reproduce memory-path outputs. Retain
  absent CRS and exact geographic grids. Persistence is optional for callers.
- [x] Measure wall time and allocated accumulator bytes at parent shapes 256x256,
  512x512, and 1024x1024 with fixed channels/recipe. Report full-parent buffers
  honestly; do not call them bounded spatial chunking. A measured unsuitable
  memory profile requires a separate out-of-core backend design before scale
  claims, not silent whole-raster eager execution.
- [x] Run `uv run --no-sync python docs/superpowers/probes/2026-10-04-merger-comparison.py`.
  Require all supported semantics to pass or document a candidate's rejection.
- [x] Select the smallest implementation satisfying the tested contract. Retain
  Tiler window generation if it still earns its dependency; remove `tiler`
  only if both layout and accumulation no longer use it.
- [x] Write the backend-specific migration plan with exact internals/tests from
  this result. The target public contract below changes only by an explicit
  documented reason; no generic output dispatch emerges from this task.

Deliverable: one measured backend decision and its executable migration plan.
The earlier full-flow experiment does not mark these comparison steps complete.

## Task 4: Atomic dense consumer migration

**Activation:** After Task 3's decision and implementation plan are reviewed.
These are required interfaces and acceptance criteria for that plan.

**Files:** Existing stale-consumer inventory in the
[reference plan](2026-10-04-indexed-tile-reference.md#required-follow-up-atomic-consumer-migration-and-stale-code-removal),
plus `geodata/transform/tiling.py` (or a focused numeric assembly file if the
selected implementation needs it). No automatic compatibility alias.

**Interfaces:**

- Replace `TileMerger(tiles, ...)` with the spec's reference/parents constructor,
  string-keyed `add`, completed-parent `merge`, all-parent `pending`, `finish`.
- `TileDataset(tiles: Tiles, spec: ModelSpec | None = None, *,
  parent_ids: Sequence[str])`; `.reference` is generated once from its cut;
  `__getitem__(position) -> tuple[dict[str, Any], str]`.
- Supervised `Dataset` keeps manifest/spec constructor; exposes `.reference`
  and `.parents: Mapping[str, xr.DataTree]` owned by the reading process;
  `__getitem__ -> tuple[dict[str, torch.Tensor], torch.Tensor, str, torch.Tensor]`
  with an explicit spatial validity mask.
- No frames: parent ID is manifest `id`. Frames: parent ID is
  `f"{source_id}/frame-{frame_index}"` within the reference snapshot.
  Validate generated parent IDs are unique. Do not parse IDs for coordinates.
- Dataset `.merger()` constructs the selected reference merger using a native
  parent raster from the shared stack grid. `Module._merge` consumes string IDs
  and retrieves labels directly from `.parents[parent_id][target].dataset`.

- [x] Preserve manifest/frame identity before flattening, and reproduce the
  same keys in reopened workers. Serialization drops open readers, cuts, and
  prepared parent stacks; metadata references contain no open file handles.
- [x] Test equal spatial footprints at different source times/frame windows
  remain separate. Reordered manifest rows retain parent-local tile IDs.
- [x] Remove target numeric assembly. Read parent labels through the same
  float-nodata to ignore-index conversion as tile labels, assert matching target
  grids/shapes, and score each completed parent once. Test categorical values
  such as 2 and 4 remain exactly 2 and 4 rather than a blended class 3.
- [x] Keep tile targets for losses and callback outputs. Preserve Kornia crop
  behavior and unchanged ID lineage; evaluation never uses augmented geometry.
- [x] Test interrupted/sanity/limited validation scores no incomplete parent;
  a new epoch resets accumulators; full prediction requires completion.
- [x] Preserve location/time encoder inputs, exact original coordinates, nodata,
  dimensions, and multi-leading-axis results. New key collation must work with
  ordinary DataLoader and Lightning batch transfer.
- [x] Remove old live-cut merger routes, ordinal logit/target pairing, `.tolist()`
  key assumptions, and superseded tests/examples. Keep internal cutting helpers
  and native geodata mosaic/concat/merge operations that still have callers.
- [x] Run all affected tiling, dataset, supervised, callback, calibration,
  encoder-context, and spec tests, scoped lint/type checks, docs build, and a
  generated template configuration smoke. Search every live consumer for stale
  routes. Do not change calibration's tile weighting policy implicitly.

Deliverable: the reference drives dense inference and full-parent validation;
the obsolete merger path is gone. No permanent dual routing implementation.

## Task 5: First completed prediction jobs

**Activation:** After dense migration. This is a separate workflow design/probe
slice, not an instruction to create all flow skeletons now.

- [ ] Start with `predict_dense` caller/maintainer code over a released tiny
  model and a real raster. Exercise nonempty spec transforms, raw-logit assembly,
  post-assembly decode, parent completion, and geographic/pixel-only persistence.
- [ ] Add a second concrete consumer with a native decoded detector. Pin tile
  box convention, clipping, parent-scoped class-aware suppression, and an entire
  all-empty run persisted as completed. Do not reinterpret raw GeoSave head
  branch outputs without settling its decoder first.
- [ ] Compare completed family jobs against independent saved inference/assembly
  jobs using these two working consumers. Extract only the shared batch execution
  that both repeat. Demonstrate mixed inference executes the encoder once.
- [ ] Write exact workflow/config/CLI signatures after those probes; no scalar
  task selector or model-spec output registry is assumed by this plan.

Deliverable: tested workflow design based on actual output semantics. Regression,
detection training, MAE methods, and HoloViz EDA remain subsequent feature work.

## Implementation evidence

Tasks 1 and 2 have regression tests for native terminal containers, mixed
outputs/gradients, runtime annotation rejection, lazy reference construction,
exact geographic grids, absent CRS, persistence, and shuffled association for
classification and detection. Review caught optional unsupported annotation
members; per-member runtime validation and two regression cases now cover them.
The Hub wrapper also reports the generalized result type.

The preservation run passed **337 tests, 9 deselected** across chain, head,
release, tiling, GeoParquet, dataset, segmentation, encoder-context, spec, and
layering suites. The [full-flow probe](../probes/2026-10-04-predict-report.md) was
rerun over 36 tiles using the actual public reference and native terminal APIs.
The terminal probe now imports production code directly:

```bash
uv run --no-sync python docs/superpowers/probes/2026-10-04-terminal-outputs.py
```

The [backend comparison](../probes/2026-10-04-merger-comparison-report.md) covered
ten layout/window/leading-axis combinations, persisted reorder, validity,
completion, and increasing parent sizes. All candidates use full-parent buffers;
none establishes bounded spatial memory. The subsequent reference migration replaces production integer keys and
live-cut merging; its verification is recorded in the migration plan.
