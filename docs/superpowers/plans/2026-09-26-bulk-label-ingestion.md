# Bulk Label Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one-anchor ingestion with a bulk Prefect flow that discovers local label rasters, writes one self-contained label-plus-imagery Zarr per label, and publishes a portable GeoVector catalog.

**Architecture:** The flow validates primitive job inputs, deterministically expands a local label root, and fans out one deep ingestion task per label. Tasks derive native anchors internally and return completed file paths; a final task lazily registers those files with `GeoVector` and atomically writes one GeoParquet catalog.

**Tech Stack:** Python 3.12, Prefect, Pydantic, xarray/DataTree, Dask, odc-stac, GeoPandas/GeoParquet, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-bulk-label-ingestion-design.md`

## Global Constraints

- Keep `model_spec.yaml` as the only public workflow YAML; label paths, globs, destinations, and runtime source settings remain caller-owned.
- Label rasters are authoritative for sample grid and time and must have a timespan.
- Label discovery and prepared outputs are local-only; STAC catalogs and assets may be remote.
- Preserve xarray/Dask laziness until the atomic Zarr persistence step.
- Only completed sample paths cross Prefect task seams; native anchors and xarray objects remain task-local.
- GeoVector catalogs usable completed samples only; Prefect owns failures, retries, and logs.
- Do not add compatibility aliases, manifest wrappers, registries, or a second ingestion path.
- Preserve unrelated worktree changes and ignore notebooks.

## Review Focus

- An absolute glob, a parent-traversing glob, or distinct `.tif`/`.tiff` labels mapping to the same output must fail before task submission; Task 4 tests them.
- A model source named `label` must fail before any raster or STAC I/O because the prepared tree reserves that group; Task 4 tests it.
- A previously completed sample whose groups or variables no longer satisfy the current model spec must fail without overwrite; Task 2 tests it.
- A partial run must retain completed sample Zarrs but leave an existing catalog untouched; Task 4 tests it.
- Catalog construction must copy metadata without computing any Dask pixel graph; Task 3 tests it.

## File Structure

- `workflow/tasks/load.py` keeps ordinary STAC source loading and endpoint fallback; it accepts native `GeoAnchor`.
- `workflow/tasks/save.py` keeps ordinary atomic `write_stack` persistence; its transitional Prefect wrapper is removed with the old flow.
- `workflow/tasks/ingest.py` owns one label-to-sample operation, including reuse validation.
- `workflow/tasks/catalog.py` owns lazy GeoVector registration and GeoParquet publication.
- `workflow/flows/ingest.py` owns primitive validation, local discovery, naming, fan-out, and final catalog orchestration.
- `workflow/configs/source.py` remains the only runtime ingestion configuration module.
- `templates/boilerplate/scripts/ingest_imagery.py` becomes a thin project caller.

---

### Task 1: Make Raster Loading and Stack Persistence Native Operations

**Files:**
- Modify: `src/geosave_engine/workflow/tasks/load.py`
- Modify: `src/geosave_engine/workflow/tasks/save.py`
- Modify: `tests/workflow/tasks/test_load.py`
- Modify: `tests/workflow/tasks/test_save.py`

**Interfaces:**
- Consumes: `SourceConfig`, `RasterRequirement`, native `GeoAnchor`, and named `xr.Dataset` mappings.
- Produces: `RasterLoader.load(config: SourceConfig, anchor: GeoAnchor) -> xr.Dataset` and ordinary `write_stack(rasters: dict[str, xr.Dataset], output: str | Path) -> str`. The old task wrappers delegate during migration and are deleted in Task 4.

- [ ] **Step 1: Write the failing native-anchor loader test**

Change `test_load_raster_reads_and_validates_one_local_source` to call
`RasterLoader(requirement).load(SourceConfig(), expected_anchor)` and assert
the selected variables, exact grid, Dask chunks, and one STAC request. The
production mutation caught is calling `.open()` on an already-native anchor.

- [ ] **Step 2: Run the loader test to verify RED**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_load.py::test_load_raster_reads_and_validates_one_local_source -q`

Expected: FAIL because `GeoAnchor` has no `open()` method.

- [ ] **Step 3: Change the loader interface**

In `load.py`, import `GeoAnchor`, change `RasterLoader.load` to accept it, and
pass it directly to `StacSource.load`. Temporarily retain `load_raster` for the
still-live old flow, but make that wrapper open its `AnchorConfig` before
calling the native loader. Update direct loader tests to construct or open
native anchors explicitly. Task 4 deletes the wrapper and its Prefect imports.

- [ ] **Step 4: Establish the save characterization baseline**

Run the current `save_stack` tests before refactoring. They already characterize
complete round trips, local `.zarr` validation, preservation of existing
destinations, publication-race checks, and failed-write staging cleanup.

- [ ] **Step 5: Verify the characterization baseline is GREEN**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_save.py -q`

Expected: all tests PASS before the refactor.

- [ ] **Step 6: Extract ordinary `write_stack`**

Move the characterized persistence body to
`write_stack(rasters: dict[str, xr.Dataset], output: str | Path) -> str` and
make the transitional `save_stack` Prefect wrapper return
`write_stack(rasters, output)`. Change persistence behavior tests to call
`write_stack` directly and rename their `test_save_stack_*` names to
`test_write_stack_*`; retain only the wrapper's cache-policy assertion until
Task 4 removes the old flow. This is the refactor phase of the green loader
cycle, not new persistence behavior.

- [ ] **Step 7: Verify Task 1**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_load.py tests/workflow/tasks/test_save.py -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/geosave_engine/workflow/tasks/load.py src/geosave_engine/workflow/tasks/save.py tests/workflow/tasks/test_load.py tests/workflow/tasks/test_save.py
git commit -m "refactor: make ingestion io task-local"
```

---

### Task 2: Prepare and Resume One Label Sample

**Files:**
- Create: `src/geosave_engine/workflow/tasks/ingest.py`
- Create: `tests/workflow/tasks/test_ingest.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`

**Interfaces:**
- Consumes: Task 1 `RasterLoader.load` and `write_stack`, one label path, validated source configs, model source requirements, and one deterministic output path.
- Produces: Prefect `ingest_sample(label: str | Path, sources: dict[str, SourceConfig], requirements: dict[str, RasterRequirement], output: str | Path) -> str` with `NO_CACHE` and `persist_result=False`.

- [ ] **Step 1: Write the failing real sample-ingestion test**

Add `test_ingest_sample_writes_label_and_matching_imagery`. Create a real label
GeoTIFF on the `stac_server` fixture anchor with a `numpy.datetime64` time,
call `ingest_sample.fn`, reopen the result, and assert groups
`("label", "optical")`, exact shared grid, original label values, selected
`red`/`nir` imagery, lazy reopened arrays, and one `optical` STAC request.

- [ ] **Step 2: Run the sample test to verify RED**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_ingest.py::test_ingest_sample_writes_label_and_matching_imagery -q`

Expected: FAIL because `workflow.tasks.ingest` does not exist.

- [ ] **Step 3: Implement `ingest_sample` minimally**

Open the label with `io.read_raster` inside the task, require
`label.gs.anchor.timespan`, load every source with
`RasterLoader(requirement).load(source, anchor)`, and call
`write_stack({"label": label, **rasters}, output)` before the label context
closes. Add source-name notes to raised errors. Export `ingest_sample`.

- [ ] **Step 4: Verify the new sample passes**

Run the test from Step 2.

Expected: PASS.

- [ ] **Step 5: Write failing resume-validation tests**

Add:

- `test_ingest_sample_reuses_a_valid_completed_sample_without_stac`, which
  calls the task twice and asserts the second call returns the same path with
  no additional STAC request;
- `test_ingest_sample_rejects_an_existing_sample_with_missing_groups`, which
  writes a Zarr missing `label` and asserts a clear `ValueError` without
  changing it;
- `test_ingest_sample_rejects_stale_source_variables`, which writes the right
  groups but omits a required variable and asserts failure without STAC;
- `test_ingest_sample_requires_label_time`, which uses a timeless GeoTIFF and
  asserts failure before STAC.

- [ ] **Step 6: Run resume tests to verify RED**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_ingest.py -q`

Expected: new resume and timeless-label tests FAIL.

- [ ] **Step 7: Implement existing-sample validation**

Add private
`_validate_sample(path: Path, requirements: dict[str, RasterRequirement]) -> None`.
Open the stack, require exactly `{"label", *requirements}`, require a label
timespan, verify the shared grid through the stack accessor, and run each
requirement's `select_raster` against its stored source group. In
`ingest_sample`, call it and return the existing path before opening a STAC
client. Never delete or overwrite an invalid existing sample.

- [ ] **Step 8: Verify Task 2**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_ingest.py tests/workflow/tasks/test_load.py tests/workflow/tasks/test_save.py -q`

Expected: all tests PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/geosave_engine/workflow/tasks/ingest.py src/geosave_engine/workflow/tasks/__init__.py tests/workflow/tasks/test_ingest.py
git commit -m "feat: prepare one label sample"
```

---

### Task 3: Publish Completed Samples as a GeoVector Catalog

**Files:**
- Create: `src/geosave_engine/workflow/tasks/catalog.py`
- Create: `tests/workflow/tasks/test_catalog.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`

**Interfaces:**
- Consumes: insertion-ordered `dict[sample_id, completed_zarr_path]` later supplied by Task 4, plus native `GeoVector.from_xarray`, `GeoVector.concat`, and GeoParquet persistence.
- Produces: Prefect `save_catalog(samples: dict[str, str], destination: str | Path) -> str` with `NO_CACHE` and `persist_result=False`.

- [ ] **Step 1: Write the failing catalog tests**

Add `test_save_catalog_registers_completed_samples`. Persist two real sample
stacks beneath `prepared/samples`, call `save_catalog.fn`, reopen the
GeoParquet with `io.read_vector`, and assert ordered `sample_id` values,
resolved usable sample paths, geometry, time, grid metadata, and grouped
variables including `label/class` and `optical/red`.

Also add `test_save_catalog_does_not_compute_sample_pixels` using a Dask
callback around catalog creation, and
`test_save_catalog_replaces_rows_atomically`, which writes a two-row catalog
then a one-row catalog and asserts only the new row remains with no staging
file.

- [ ] **Step 2: Run the catalog test to verify RED**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_catalog.py::test_save_catalog_registers_completed_samples -q`

Expected: FAIL because `save_catalog` does not exist.

- [ ] **Step 3: Implement `save_catalog` minimally**

For each mapping entry in insertion order, open the stack with
`io.read_stack`, call `GeoVector.from_xarray` with
`fields=("time", "grid", "variables")`, `path=sample_path`, and the explicit
`sample_id`, then close it. Concatenate the records, create the destination
parent, write GeoParquet with `overwrite=True`, and return the destination
string. Use the existing atomic GeoParquet replacement rather than adding
catalog state. Set `NO_CACHE` and `persist_result=False`.

- [ ] **Step 4: Verify the catalog behaviors pass**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_catalog.py -q`

Expected: all tests PASS, including task metadata assertions.

- [ ] **Step 5: Verify Task 3**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_catalog.py tests/geodata/core/test_vector.py tests/geodata/utils/io/test_geoparquet.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/geosave_engine/workflow/tasks/catalog.py src/geosave_engine/workflow/tasks/__init__.py tests/workflow/tasks/test_catalog.py
git commit -m "feat: publish ingestion catalog"
```

---

### Task 4: Replace One-Anchor Ingestion with the Bulk Flow

**Files:**
- Modify: `src/geosave_engine/workflow/flows/ingest.py`
- Modify: `tests/workflow/flows/test_ingest.py`
- Modify: `src/geosave_engine/workflow/configs/__init__.py`
- Delete: `src/geosave_engine/workflow/configs/anchor.py`
- Delete: `src/geosave_engine/workflow/configs/ingest.py` if still present in the worktree/index
- Delete: `tests/workflow/configs/test_anchor.py`
- Delete: `tests/workflow/configs/test_ingest.py` if still present in the worktree/index

**Interfaces:**
- Consumes: Task 2 `ingest_sample`, Task 3 `save_catalog`, `SourceConfig`, and `ModelSpec`.
- Produces: Prefect `ingest(labels: str, sources: dict[str, dict[str, JsonValue]], *, output: str, spec: str, pattern: str = "**/*.tif") -> str` returning `<output>/manifest.parquet`.

- [ ] **Step 1: Write failing discovery tests**

Replace the one-anchor flow tests with tests that call `ingest.fn` and assert:

- a missing or non-directory label root fails;
- an absolute glob fails;
- a glob containing a `..` segment fails rather than discovering outside the label root;
- a glob with no matching files fails;
- a directory matched as a label fails;
- `a.tif` and `a.tiff` matched by `**/*.tif*` fail because both map to
  `samples/a.zarr`.

Patch only task submission in these preflight tests so any submission raises;
assert the public validation error rather than mock call counts.

- [ ] **Step 2: Run discovery tests to verify RED**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/flows/test_ingest.py -q -k 'root or glob or collision'`

Expected: FAIL because the current flow still requires one `anchor`.

- [ ] **Step 3: Implement deterministic local discovery**

Add private
`_discover_labels(root: Path, pattern: str) -> dict[str, Path]` in the flow
module. Return sorted relative POSIX IDs mapped to paths, validate every
preflight condition above, and detect output collisions after replacing each
suffix with `.zarr`.

- [ ] **Step 4: Write failing flow-configuration tests**

Retain nested `SourceConfig` validation before task submission. Add failures
for missing/extra source names, no model sources, a reserved model source
named `label`, a URI output, and an output path that is an existing file.

- [ ] **Step 5: Implement the bulk flow orchestration**

Parse all source settings with `SourceConfig.model_validate`, load
`ModelSpec`, validate exact source names and the reserved group, discover
labels, and derive each path as
`Path(output) / "samples" / Path(sample_id).with_suffix(".zarr")`. Submit one
`ingest_sample` per label, resolve paths in discovery order, then submit
`save_catalog` to `<output>/manifest.parquet` and return its result.

- [ ] **Step 6: Write the failing two-label integration and resume test**

Using the real `stac_server` and Prefect harness, write two timed label
GeoTIFFs under `labels/train`, run the flow, and assert:

- two mirrored sample Zarrs contain label and imagery groups;
- the returned manifest contains two ordered rows and usable relative paths;
- running the same flow again returns the same manifest;
- the second run makes no additional STAC requests.

- [ ] **Step 7: Run the integration test to verify RED, then complete fan-out**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/flows/test_ingest.py::test_ingest_prepares_many_labels_and_resumes -q`

Expected before completing orchestration: FAIL. After the minimal fan-out and
catalog wiring: PASS.

- [ ] **Step 8: Write and pass the partial-failure test**

Add `test_ingest_keeps_completed_samples_without_replacing_manifest`. Seed an
old one-row manifest, provide a sorted timed `a.tif` and timeless `b.tif`, run
the flow expecting failure, and assert `a.zarr` exists while the old manifest
still contains its original row.

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/flows/test_ingest.py::test_ingest_keeps_completed_samples_without_replacing_manifest -q`

Expected: PASS.

- [ ] **Step 9: Remove the obsolete anchor configuration seam**

Delete the anchor config module/tests and remove their exports. Delete the
transitional `load_raster` and `save_stack` Prefect wrappers and remove their
exports; `ingest_sample` uses `RasterLoader` and `write_stack` internally.
Expose `ingest_sample` and `save_catalog` without import aliases. Update all
live source and test imports; leave historical plans/specs unchanged.

- [ ] **Step 10: Verify Task 4**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow -q`

Expected: all workflow tests PASS, with only explicitly configured deselections
and third-party warnings.

- [ ] **Step 11: Commit Task 4**

```bash
git add src/geosave_engine/workflow/flows/ingest.py src/geosave_engine/workflow/configs src/geosave_engine/workflow/tasks/__init__.py tests/workflow/flows/test_ingest.py tests/workflow/configs
git commit -m "feat: bulk ingest label datasets"
```

---

### Task 5: Replace the Generated Manifest Script and Verify the Repository

**Files:**
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Create: `tests/workflow/test_ingest_script.py`
- Modify: `README.md` only if its existing workflow description names the removed one-anchor interface

**Interfaces:**
- Consumes: Task 4 public `ingest` flow and the supervised template's `sentinel_2_l2a` model source name.
- Produces: thin Typer `main(labels: Path, output: Path, spec: Path, pattern: str) -> None` that prints the returned GeoParquet path.

- [ ] **Step 1: Write the failing thin-script test**

Add `test_ingest_script_passes_bulk_job_inputs`. Import the template module,
replace only the external Prefect flow seam with a capturing fake, invoke
`main` with literal paths/pattern, and assert the printed manifest plus this
exact source mapping:

```python
{"sentinel_2_l2a": {"query": {}, "load": {}}}
```

The production mutation caught is retaining label discovery, STAC loading, or
CSV/Excel manifest behavior in the generated script.

- [ ] **Step 2: Run the script test to verify RED**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/test_ingest_script.py -q`

Expected: FAIL because the current script implements its own ingestion and
`IngestManifest`.

- [ ] **Step 3: Replace the script with the thin caller**

Keep environment loading and GDAL retry configuration. Remove pandas, direct
STAC clients, label discovery, anchor extraction, raster writing, and
`IngestManifest`. Declare the project source mapping once, call the bulk flow,
and print its returned manifest path.

- [ ] **Step 4: Verify live references and focused quality checks**

Run:

```bash
rg -n "IngestManifest|AnchorConfig|CoordinateAnchorConfig|GeoJSONAnchorConfig|RasterAnchorConfig|load_raster|save_stack" src tests --glob '!*.ipynb'
UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow tests/geodata/core/test_vector.py tests/geodata/utils/io/test_geoparquet.py -q
UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/workflow src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py tests/workflow
UV_CACHE_DIR=/tmp/geosave-uv-cache uv run basedpyright src/geosave_engine/workflow
git diff --check
```

Expected: `rg` returns no live references; tests, Ruff, BasedPyright, and diff
check pass.

- [ ] **Step 5: Run the full suite and report unrelated failures exactly**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest`

Expected: all tests pass, or the known unrelated incomplete ML migration still
causes collection errors for deleted `geosave_engine.ml.data` and
`geosave_engine.ml.tasks`; report exact modules without changing them.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py tests/workflow/test_ingest_script.py README.md
git commit -m "refactor: use bulk ingestion in workspace script"
```
