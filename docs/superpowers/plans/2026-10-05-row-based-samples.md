# Row-based Samples Implementation Plan

> Use superpowers:executing-plans for inline implementation and a final review.

**Goal:** Separate raster reading from training sample construction and use
the same row-based context recipe in training and inference.

**Spec:** `docs/superpowers/specs/2026-10-05-row-based-samples-design.md`

**Architecture:** Native xarray, GeoDataFrame, Tiler/Merger, and method-owned
Lightning datasets. No new dependencies or compatibility paths.

## Proposed architectural revision

The original tasks below are completed historical work. Preparation-job and
whole-catalog conversion proposals are withdrawn. The current revision is
[`native raster storage and GeoVector records`](../specs/2026-10-05-native-sample-inputs-design.md).

Start from existing read_raster/read_stack, format writers, GeoVector registration,
GeoParquet persistence, and reopening selected asset records. No new job API.

1. Preserve and verify the explicit asset -> GeoVector -> Parquet -> asset path,
   including moved directories and lazy metadata registration.
2. Make format writer return values consistent. Add optional catalog/id arguments
   to existing writers, delegating record construction to GeoVector and catalog
   persistence to the existing GeoParquet implementation after assets complete.
3. Complete multi-group Zarr registration/reopening through named asset pointers
   and native group selectors. Remove the restriction requiring separate stores.
   Use the same asset/path selection rules for registration and reopening.
4. Put the selected-row reader on Series using the same core asset reader; remove
   data= substitution. Preserve generic vector selection and native record IDs.
5. Verify changed pixels register new saved assets, actual metadata round trips,
   native time coordinates, mixed formats, relocation, and partial publication.
   Keep independent-grid registration and conventional per-leaf COG cataloging
   explicit unresolved contracts rather than silently guessing their layout.

Only after this core path is sound, revisit model preprocessing, method-owned
training datasets, and Prefect composition. Do not implement from the superseded
proposal blocks below.

## Tasks

1. Add failing behavioral tests for metadata-only indexed reads, filtered
   references, ordered temporal metadata, and shuffled dense assembly.
   Replace shared TileDataset with optional-window `GeoVector.to_xarray`; rename
   the reference factory to `from_layouts`; add per-raster metadata. Route IDs
   through the reference into native Merger without a new wrapper. Retain bounded reads, halo values, exact grids and native axes.
2. Add failing tests for row-based Prithvi/Clay context, inert recipe persistence,
   explicit cache equivalence, and missing/conflicting context. Implement the
   encoder functions, ModelSpec context declaration, and ML tensor conversion.
3. Migrate supervised Dataset and existing tests; run a tiny-model Lightning
   prediction and shuffled native merge, plus real asset/worker reopening tests.
   Remove stale shared Dataset imports and unsupported context declarations.
4. Update current architecture/workspace docs and template configuration.
   Run focused regression tests, lint, changed-file type checks, docs build,
   dependency-direction checks, and diff checks. Obtain one read-only review.

## Review focus

- Persisted nested metadata must handle Parquet arrays and missing values.
- Context must preserve per-input timestamp order and distinguish tile grids.
- An explicit cached context must never overwrite raster inputs.
- Filtered/reordered references must read the same bounded pixels by ID.
- Native fringe/halo padding and coverage must preserve full-parent evaluation.

Execution stays in the user's active checkout because the preceding refactors
are already there and include uncommitted changes required by this work.


## Execution results

- Tasks 1–3: implemented in the existing GeoVector reader, ModelSpec, encoder
  modules, and supervised Dataset. No reader/merger wrapper was added.
- Initial regression: 475 passed; explicit slow worker/layer checks: 6 passed.
- Read-only review found repeated lazy reflect/wrap padding truncation. Eight
  native-parity tests failed before the fix; the padding/input suite then
  passed all 35 tests. NumPy pads one-dimensional index arrays and xarray keeps
  the pixel selections lazy.
- WKT-only regular rows gained a focused missing-code regression and fix.
- Final full focused regression, type/lint/docs/diff results are recorded in
  the execution ledger after the fix pass. Final regression: 484 passed,
  9 deselected; worker test: 1 passed; Ruff, scoped BasedPyright, Zensical,
  and diff checks passed. Three existing encoder return-type diagnostics and
  zero-tile native layouts remain outside this fix pass.


## Native storage revision execution

- Implemented optional companion catalogs on existing COG/Zarr accessors;
  Dataset COG writes now return their Path. Catalog construction uses actual
  saved assets and existing GeoVector/GeoParquet operations.
- Added a shared asset reader for registration and selected Series reopening.
  Multi-group Zarr pointers retain href/group and native inherited coordinates.
- Removed the table-level `to_xarray(id, data=...)`; training and examples select
  native rows, then open assets or explicitly crop prepared parents.
- Ruling: keep preparation, model input recipes, and Prefect redesign outside
  this persistence revision. The existing method-owned training behavior only
  changes how it opens a row and applies its explicit window.
- Watched new catalog tests fail before implementation; after fixes, 10 passed.
  Dependency findings: group-only Dataset reads lacked inherited coordinates;
  xarray's delayed write returned None. Native DataTree selection and a delayed
  saved-path result resolve both without recreating metadata.
- Added relocation, changed-pixel, GeoJSON without ID, failure, overwrite,
  local-URI, and deferred-publication coverage. Final checks follow below.

- First full default regression: 1,378 passed, 25 deselected. Explicit slow
  worker/import checks: 6 passed, 17 deselected. Ruff, scoped BasedPyright,
  Zensical, and diff checks passed.
- Read-only review found file-closure loss through native stack construction and
  a dependency minimum below the group coordinate API. Watched three closure
  tests fail, then all 14 asset/row tests pass after attaching native close
  callbacks and closing reads on crop failure. Reviewer verified both fixes.
- Ruling: require xarray >= 2026.4.0 rather than recreate inherited coordinates
  for older versions. This is the native API used by group reads; the lockfile
  already resolved that version. Lock refresh and offline lock check passed.

- Final default regression after review fixes: 1,383 passed, 25 deselected.
  Scoped BasedPyright: zero diagnostics. Ruff checks and formatting passed;
  Zensical build, offline lock consistency, and git diff checks passed.
  No remaining material review findings. No wheels, commits, or publishing.
