# Dense Workflow Namespace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose dense training preparation through the concise `workflow.dense` namespace while keeping generic raster workflow operations separate.

**Architecture:** One `workflow/dense.py` module owns label discovery, dense sample validation, its private per-label Prefect task, and the public `prepare` flow. Generic acquisition and persistence remain in `workflow.flows.ingest` and `workflow.tasks`; no task registry or compatibility alias is added.

**Tech Stack:** Python 3.12, Prefect, Pydantic, xarray/DataTree, GeoVector/GeoParquet, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-ingest-prepare-split-design.md`

## Global Constraints

- Public usage is `from geosave_engine.workflow import dense`, followed by `dense.prepare(...)` or `dense.validate_sample(...)`.
- Keep `model_spec.yaml` model-owned; labels, paths, outputs, STAC bindings, and Prefect settings remain caller-owned.
- Pass completed paths, not open rasters or clients, across Prefect boundaries.
- Keep source loading lazy until `write_stack` persists the completed sample.
- Remove obsolete APIs without aliases or parallel execution paths.
- Preserve unrelated worktree changes.

## Review Focus

- Invalid source bindings must fail before label discovery or pixel work.
- A valid existing sample must be reused without another STAC search.
- A failed sample must not replace an existing manifest.
- A configured source concurrency limit must cover lazy loading through sample persistence.
- Old `prepare_training`, `write_sample`, and `prepare_sample` imports must not survive.

---

### Task 1: Consolidate Dense Preparation

**Files:**
- Create: `src/geosave_engine/workflow/dense.py`
- Modify: `src/geosave_engine/workflow/__init__.py`
- Modify: `src/geosave_engine/workflow/flows/__init__.py`
- Delete: `src/geosave_engine/workflow/flows/prepare_training.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`
- Delete: `src/geosave_engine/workflow/tasks/sample.py`
- Create: `tests/workflow/test_dense.py`
- Delete: `tests/workflow/flows/test_prepare_training.py`
- Delete: `tests/workflow/tasks/test_sample.py`
- Modify: `tests/workflow/tasks/test_catalog.py`

**Interfaces:**
- Consumes: `load_raster(anchor, source, requirement) -> xr.Dataset`, `source_concurrency(sources)`, `write_stack(rasters, destination) -> str`, and `write_manifest(samples, destination) -> str`.
- Produces: `dense.prepare(labels: str, sources: dict[str, dict[str, JsonValue]], *, output: str, spec: str, pattern: str = "**/*.tif") -> str` and `dense.validate_sample(path: str | Path, requirements: dict[str, RasterRequirement]) -> None`.

- [ ] **Step 1: Write the failing public-namespace test**

  Add `test_dense_exposes_only_prepare_and_validate_sample` to `tests/workflow/test_dense.py`. Import `dense` from `geosave_engine.workflow`, assert `dense.prepare` is a Prefect `Flow`, assert both public callables exist, and assert `prepare_training`, `write_sample`, and `prepare_sample` are absent from their former package exports.

- [ ] **Step 2: Run the namespace test to verify it fails**

  Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/test_dense.py::test_dense_exposes_only_prepare_and_validate_sample -v`

  Expected: FAIL because `geosave_engine.workflow.dense` does not exist.

- [ ] **Step 3: Move the dense behavior tests under their new owner**

  Merge the behavioral coverage from `test_prepare_training.py` and `test_sample.py` into `test_dense.py`. Rename tests around the public `dense.prepare` and `dense.validate_sample` interfaces while retaining exact assertions for label-derived anchors, resume without STAC, source binding validation, output collisions, missing label time, partial failure, manifest rows, and concurrency limits.

- [ ] **Step 4: Implement the cohesive dense module**

  Create `workflow/dense.py` with these exact definitions:

  - `validate_sample(path: str | Path, requirements: dict[str, RasterRequirement]) -> None`
  - private `_discover_labels(root: Path, pattern: str) -> dict[str, Path]`
  - private Prefect task `_prepare_sample(label: str | Path, sources: dict[str, SourceConfig], requirements: dict[str, RasterRequirement], output: str | Path) -> str`
  - public Prefect flow `prepare(labels, sources, *, output, spec, pattern="**/*.tif") -> str`, named `dense-prepare`

  Co-locate the existing logic without introducing a protocol, class, registry, or compatibility wrapper. Export the `dense` module from `workflow/__init__.py`; remove the old flow and task exports and files. Rename the local `write_sample` fixture helper in `test_catalog.py` so it cannot be mistaken for a surviving public API.

- [ ] **Step 5: Run dense and generic-ingest tests**

  Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/test_dense.py tests/workflow/flows/test_ingest.py -m "not integration"`

  Expected: all tests PASS, including the slow Prefect cases when local socket access is available.

- [ ] **Step 6: Confirm obsolete names are gone**

  Run: `rg -n "prepare_training|write_sample|prepare_sample" src/geosave_engine tests/workflow --glob '*.py'`

  Expected: no matches.

- [ ] **Step 7: Commit the namespace migration**

  ```bash
  git add src/geosave_engine/workflow tests/workflow
  git commit -m "refactor: namespace dense preparation"
  ```

### Task 2: Update the Generated Workspace and Verify

**Files:**
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Modify: `tests/workflow/test_ingest_script.py`

**Interfaces:**
- Consumes: `dense.prepare(...) -> str` from Task 1.
- Produces: the existing Typer script behavior, printing the returned manifest path.

- [ ] **Step 1: Write the failing workspace test**

  Update `test_ingest_script_calls_dense_prepare` to replace `ingest_imagery.dense.prepare`, call `main`, and assert the same serialized label, source, output, spec, and pattern arguments plus printed manifest path.

- [ ] **Step 2: Run the workspace test to verify it fails**

  Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/test_ingest_script.py -v`

  Expected: FAIL because the script still imports `prepare_training`.

- [ ] **Step 3: Use the dense namespace in the template**

  Import `dense` from `geosave_engine.workflow` and replace the old flow call with `dense.prepare`. Keep project defaults, dotenv loading, GDAL configuration, and output unchanged.

- [ ] **Step 4: Run workflow verification**

  Run:

  ```bash
  UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow -m "not integration"
  UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest
  UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/workflow src/geosave_engine/templates tests/workflow
  UV_CACHE_DIR=/tmp/geosave-uv-cache uv run basedpyright src/geosave_engine/workflow tests/workflow
  git diff --check
  ```

  Expected: workflow tests, Ruff, BasedPyright, and diff validation PASS. If the full repository suite is still blocked by unrelated ML collection errors, report those without changing ML code.

- [ ] **Step 5: Commit the workspace migration**

  ```bash
  git add src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py tests/workflow/test_ingest_script.py docs/superpowers/plans/2026-09-26-dense-workflow-namespace.md
  git commit -m "refactor: prepare dense data from workspace"
  ```
