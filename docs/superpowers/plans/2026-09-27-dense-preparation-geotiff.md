# Dense Preparation GeoTIFF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dense preparation publish flat, discoverable COG samples by default, retain explicit Zarr output, and preserve source-relative dataset splits.

**Architecture:** A shared sample-persistence seam in `workflow.tasks.save` writes and reopens either a flat GeoTIFF sample directory or a grouped Zarr store. The dense task, catalog task, flow, CLI, and generated script pass one literal format through that seam; prediction separately replaces its generator context manager with explicit `ExitStack` ownership.

**Tech Stack:** Python 3.12, Pydantic 2, xarray/DataTree, Dask, Rasterio/GDAL COG, Zarr, GeoParquet, Prefect 3, Typer, pytest, Ruff, BasedPyright

**Spec:** `docs/superpowers/specs/2026-09-27-dense-preparation-geotiff-design.md`

## Global Constraints

- `format` is `Literal["geotiff", "zarr"]` and defaults to `"geotiff"` on `prepare_dense_data`.
- GeoTIFF means one flat multiband COG per named raster inside one atomic sample directory.
- A GeoTIFF raster accepts no time dimension or exactly one time step; multiple steps require Zarr.
- The output mirrors the suffix-free label path with no intermediate `samples/` directory.
- `write_options` passes serializable encoding options to native writers; layout, band splitting, overwrite, and deferred computation remain workflow-owned.
- Keep `model_spec.yaml` free of job-owned output format and encoding settings.
- Do not change the general `GeoStack.to_cog()` layout.
- Do not add compatibility aliases or a legacy output path.
- Preserve unrelated working-tree changes. Several target files are already modified or untracked; inspect every diff and do not commit pre-existing user work.
- Public APIs use concise Google-style docstrings; private helpers have short docstrings.
- Ignore notebooks.

## Review Focus

- A multi-suffix label such as `tile.v1.tif` must become `tile.v1/` or `tile.v1.zarr`, not `tile/`; Task 3 adds the path test.
- A GeoTIFF destination that is a file, or a sample directory with missing or extra assets, must fail without overwrite; Tasks 1 and 2 add persistence and resume tests.
- `write_options` containing `layout`, `split_bands`, `overwrite`, or `compute` must fail before writing; Task 1 adds the reserved-option test.
- A singleton time dimension must reopen as the same scalar timestamp, while two time steps must fail with guidance to use Zarr; Task 1 adds both tests.
- A failure after one COG is written must remove staging and leave no final sample directory; Task 1 adds a delayed-pixel failure test.

---

### Task 1: Native prepared-sample persistence

**Files:**
- Modify: `src/geosave_engine/workflow/tasks/save.py`
- Modify: `tests/workflow/tasks/test_save.py`

**Interfaces:**
- Consumes: `stack(rasters: Mapping[str, xr.Dataset]) -> xr.DataTree`, `io.geotiff.write_cog`, `io.read_raster`, `io.read_stack`, and the existing atomic `write_stack` behavior.
- Produces: `SampleFormat = Literal["geotiff", "zarr"]`; `open_sample(source: str | Path, *, format: SampleFormat) -> xr.DataTree`; `write_sample(rasters: dict[str, xr.Dataset], output: str | Path, *, format: SampleFormat = "geotiff", write_options: Mapping[str, JsonValue] | None = None) -> str`.

- [ ] **Step 1: Write failing flat-GeoTIFF and reopen tests**

Add `test_write_sample_publishes_flat_geotiff_assets` and `test_open_sample_rebuilds_the_logical_stack`. Write `label` and `optical` datasets with different dtypes, assert the result is the sample directory, assert its files are exactly `label.tif` and `optical.tif`, then reopen and assert groups, variables, shared grid, and pixel values.

- [ ] **Step 2: Write failing time, option, destination, and atomicity tests**

Add tests that assert:

- a one-element `time` dimension reopens as the same scalar timestamp;
- two time steps raise `ValueError` matching `use format='zarr'`;
- each of `layout`, `split_bands`, `overwrite`, and `compute` raises before a writer is called;
- an existing file at the sample-directory path raises `FileExistsError`; and
- a delayed failure in the second raster leaves no destination and no `.<name>-*` staging directory.

- [ ] **Step 3: Write a failing explicit-Zarr and option-forwarding test**

Add `test_write_sample_zarr_round_trip` and a monkeypatched COG test asserting `{"compress": "ZSTD", "blocksize": 256}` reaches `io.geotiff.write_cog` unchanged while GeoSave supplies the path and overwrite behavior.

- [ ] **Step 4: Run the persistence tests to verify the red state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_save.py -q`

Expected: FAIL because `open_sample`, `write_sample`, and `SampleFormat` do not exist.

- [ ] **Step 5: Implement the sample persistence seam**

In `save.py`, define the interfaces above. For GeoTIFF, reduce only a one-element time dimension with `drop=False`, stage a sibling sample directory, write `<name>.tif` with `io.geotiff.write_cog`, reopen the staged files as a `DataTree`, and rename only after success. For Zarr, preserve the existing atomic stack writer and pass allowed native write options through with eager computation fixed by GeoSave.

- [ ] **Step 6: Run the persistence tests to verify the green state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_save.py tests/geodata/utils/io/test_geotiff.py tests/geodata/utils/io/test_zarr_stack.py -q`

Expected: PASS.

- [ ] **Step 7: Inspect and checkpoint Task 1**

Run: `git diff --check -- src/geosave_engine/workflow/tasks/save.py tests/workflow/tasks/test_save.py`

Inspect the complete diff. Commit only if the staged patch contains no pre-existing user work; otherwise leave the verified changes uncommitted and record the overlap.

### Task 2: Dense task and manifest integration

**Files:**
- Modify: `src/geosave_engine/workflow/tasks/dense.py`
- Modify: `src/geosave_engine/workflow/tasks/catalog.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`
- Modify: `tests/workflow/tasks/test_dense.py`
- Modify: `tests/workflow/tasks/test_catalog.py`

**Interfaces:**
- Consumes: Task 1 `SampleFormat`, `open_sample`, and `write_sample`; existing `load_stac_raster`; `RasterRequirement.validate_raster`.
- Produces: public Prefect task `prepare_dense_sample(label: str | Path, requirements: dict[str, RasterRequirement], output: str | Path, *, format: SampleFormat = "geotiff", write_options: Mapping[str, JsonValue] | None = None) -> str`; `write_manifest(samples: dict[str, str], destination: str | Path, *, format: SampleFormat) -> str`.

- [ ] **Step 1: Update dense-task tests to the public, format-aware contract**

Rename test calls from `_prepare_dense_sample` to `prepare_dense_sample`. Make GeoTIFF the default assertion, add explicit Zarr coverage for the old grouped-store behavior, and assert a valid existing GeoTIFF sample is reused without STAC requests.

- [ ] **Step 2: Add invalid GeoTIFF resume tests**

Add parameterized coverage for a missing required asset, an unexpected `.tif` asset, a mismatched grid, and a stale raster variable. Each case must raise without calling STAC or modifying the destination.

- [ ] **Step 3: Update catalog tests for both persisted representations**

For GeoTIFF and Zarr samples, assert `write_manifest` records suffix-free IDs, the sample root/store path, `format`, grouped variables, geometry, time, and grid fields without computing pixels during catalog construction.

- [ ] **Step 4: Run dense and catalog tests to verify the red state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_dense.py tests/workflow/tasks/test_catalog.py -q`

Expected: FAIL because the task remains private and both modules assume Zarr.

- [ ] **Step 5: Implement format-aware validation and public task ownership**

Rename the Prefect task to `prepare_dense_sample`, export it from `workflow.tasks`, and remove the private cross-module seam. Keep `_validate_dense_sample` local; open samples through Task 1, require the exact expected group set and shared anchor, validate label time and every model raster, then call `write_sample` with the literal format and options.

- [ ] **Step 6: Make manifest publication representation-independent**

Add required keyword `format` to `write_manifest`, reopen each path through `open_sample`, and pass `format=format` into `GeoVector.from_xarray` alongside the stable sample fields.

- [ ] **Step 7: Run dense and catalog tests to verify the green state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_dense.py tests/workflow/tasks/test_catalog.py tests/workflow/tasks/test_save.py -q`

Expected: PASS.

- [ ] **Step 8: Inspect and checkpoint Task 2**

Run: `git diff --check -- src/geosave_engine/workflow/tasks tests/workflow/tasks`

Inspect the complete diff. Commit only patches that do not absorb pre-existing user work.

### Task 3: Flow, CLI, generated script, and user documentation

**Files:**
- Modify: `src/geosave_engine/workflow/flows/prepare_dense_data.py`
- Modify: `src/geosave_engine/cli/commands/workflow.py`
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Modify: `tests/workflow/flows/test_prepare_dense_data.py`
- Modify: `tests/cli/commands/test_workflow.py`
- Modify: `tests/workflow/test_ingest_script.py`
- Modify: `docs/guides/workflows.md`

**Interfaces:**
- Consumes: Task 2 `prepare_dense_sample` and format-aware `write_manifest`.
- Produces: `prepare_dense_data(..., format: Literal["geotiff", "zarr"] = "geotiff", write_options: dict[str, JsonValue] | None = None) -> str`; CLI `--format` and `--write-options JSON`; documented mirrored dataset layout.

- [ ] **Step 1: Write failing label-discovery and output-path tests**

Replace source-suffix sample assertions with suffix-free IDs. Assert `labels/train/region/tile.v1.tif` maps to GeoTIFF directory `prepared/train/region/tile.v1` and explicit Zarr store `prepared/train/region/tile.v1.zarr`. Retain the `a.tif`/`a.tiff` collision failure and stable sorted order.

- [ ] **Step 2: Update flow behavior tests for the GeoTIFF default**

Change the real slow test to assert flat `label.tif` and `optical.tif` assets under mirrored split directories, a `format == "geotiff"` manifest column, suffix-free IDs, and reuse without new STAC requests. Add one focused explicit-Zarr flow test and update concurrency/failure fakes to receive format and write options.

- [ ] **Step 3: Write failing CLI and generated-script forwarding tests**

Assert help shows `--format`, `--write-options`, and default `geotiff`; valid JSON options and `zarr` reach the flow unchanged; an unknown literal or non-object JSON exits before invoking the flow. Update the generated-script test to assert its default GeoTIFF format and optional writer settings.

- [ ] **Step 4: Run the public-surface tests to verify the red state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py tests/workflow/test_ingest_script.py -q`

Expected: FAIL because discovery embeds the source suffix and the public surfaces lack format settings.

- [ ] **Step 5: Implement suffix-free discovery and literal output routing**

Replace `_discover_labels` with `_find_labels`. Build each GeoTIFF path by joining the suffix-free relative identity directly; build Zarr by appending `.zarr` to the final name rather than using `Path.with_suffix`. Import public `prepare_dense_sample`, keep bounded submission unchanged, and pass format to manifest publication.

- [ ] **Step 6: Implement CLI and template parameters**

Annotate CLI format with `Literal["geotiff", "zarr"]`, parse `--write-options` through `TypeAdapter(dict[str, JsonValue])`, and forward both settings. Give the generated script a matching literal format plus a JSON string option parsed through the same mapping type, and retain concise Google-style docstrings.

- [ ] **Step 7: Update the workflow guide**

Document the GeoTIFF default, flat per-sample assets, mirrored split paths, single-scene restriction, explicit `--format zarr`, JSON writer options, suffix-free manifest IDs, and the Python equivalent. Remove statements that prepared output is always beneath `samples/` or always Zarr.

- [ ] **Step 8: Run the public-surface tests to verify the green state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py tests/workflow/test_ingest_script.py -q`

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py -m slow -q`

Expected: PASS.

- [ ] **Step 9: Inspect and checkpoint Task 3**

Run: `git diff --check -- src/geosave_engine/workflow/flows/prepare_dense_data.py src/geosave_engine/cli/commands/workflow.py src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py tests/workflow/test_ingest_script.py docs/guides/workflows.md`

Inspect the complete diff. Commit only patches that do not absorb pre-existing user work.

### Task 4: Prediction resource ownership

**Files:**
- Modify: `src/geosave_engine/workflow/flows/predict.py`
- Modify: `tests/workflow/flows/test_predict.py`

**Interfaces:**
- Consumes: `contextlib.ExitStack`, `io.read_stack`, and `io.read_raster`.
- Produces: `_open_rasters(source: str | Path, spec: ModelSpec, resources: ExitStack) -> Mapping[str, xr.Dataset | xr.DataArray | xr.DataTree]`; unchanged public `predict` signature and behavior.

- [ ] **Step 1: Write the failing lifetime test**

Add a raster with `set_close` callback, monkeypatch `io.read_raster`, call `_open_rasters` inside an `ExitStack`, and assert the raster is open inside the block and closed after exit. Keep the existing real raster and ingest-anchor prediction tests unchanged.

- [ ] **Step 2: Run prediction tests to verify the red state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/flows/test_predict.py -q`

Expected: FAIL because `_open_rasters` is itself a generator context manager and does not accept resource ownership.

- [ ] **Step 3: Replace the generator context manager with `ExitStack`**

Remove `contextmanager` and `Iterator`. Make `_open_rasters` an ordinary function taking `resources`, register the selected DataTree or Dataset with `resources.enter_context`, and retain the existing single-raster requirement validation and stack fallback behavior. Own one `ExitStack` around preprocessing, prediction, and export creation in `predict`.

- [ ] **Step 4: Run prediction tests to verify the green state**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/flows/test_predict.py tests/workflow/test_prediction.py tests/workflow/tasks/test_predict.py -q`

Expected: PASS.

- [ ] **Step 5: Inspect and checkpoint Task 4**

Run: `git diff --check -- src/geosave_engine/workflow/flows/predict.py tests/workflow/flows/test_predict.py`

Inspect the complete diff. Commit only patches that do not absorb pre-existing user work.

### Task 5: Integrated verification

**Files:**
- Verify only; modify earlier task files only when a failing check exposes a defect in this plan's scope.

**Interfaces:**
- Consumes: Tasks 1-4 completed implementation.
- Produces: Evidence that the changed workflow, persistence, CLI, template, and documentation surfaces agree.

- [ ] **Step 1: Run the complete affected default suite**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/geodata/utils/io tests/workflow tests/cli -q`

Expected: PASS.

- [ ] **Step 2: Run slow dense preparation coverage**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py -m slow -q`

Expected: PASS.

- [ ] **Step 3: Run scoped static checks**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync ruff check src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow.py src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py tests/workflow tests/cli/commands/test_workflow.py`

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync basedpyright src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow.py tests/workflow tests/cli/commands`

Expected: no diagnostics.

- [ ] **Step 4: Build documentation and inspect the final patch**

Run: `UV_CACHE_DIR=/tmp/geosave-dense-geotiff-cache uv run --offline --no-sync zensical build --clean`

Run: `git diff --check`

Search: `rg -n "_prepare_dense_sample|_discover_labels|samples/.*\\.zarr|sample Zarrs|contextmanager" src tests docs/guides --glob '!*.ipynb'`

Expected: documentation builds, the diff has no whitespace errors, and no obsolete live API or prepared-layout reference remains.

- [ ] **Step 5: Report the dirty-worktree boundary**

List files changed for this feature, checks run, breaking changes, and any relevant pre-existing modifications that prevented isolated task commits. Do not include or rewrite unrelated user changes.
