# Workflow Layering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the job-domain workflow packages with stable `configs`, `flows`, and `tasks` layers while preserving current ingestion and dense-preparation behavior.

**Architecture:** Runtime schemas live in `workflow.configs`, independently runnable Prefect entry points live in `workflow.flows`, and reusable work plus its cohesive storage operations live in `workflow.tasks`. `ModelSpec` continues to own raster acquisition and preprocessing; this migration changes ownership and imports only.

**Tech Stack:** Python 3.12, Pydantic 2, Prefect 3, xarray/DataTree, GeoPandas, pytest, Ruff, BasedPyright, Hatchling

**Spec:** `docs/superpowers/specs/2026-09-30-workflow-layering-design.md`

## Global Constraints

- Preserve the current ingestion, dense preparation, metadata, resume, bounded-concurrency, laziness, and atomic-publication behavior.
- Keep `ModelSpec` as the owner of raster acquisition and preprocessing declarations.
- Do not restore removed prediction, postprocessing, or workflow source-configuration APIs.
- Do not add compatibility aliases for `workflow.ingestion` or `workflow.training_data`.
- Keep only independently runnable jobs as flows.
- Keep `prepare_dense_sample` as the only submitted concurrency-bound task; do not decorate storage helpers merely because they live in `workflow.tasks`.
- Do not add `ModelSpec.preprocess()` to dense preparation in this migration.
- Preserve unrelated dirty-worktree changes and stage only task-owned paths.

## Review Focus

- A missing raster recipe must fail before opening a remote raster anchor or contacting STAC; Task 1 keeps the existing no-I/O assertion.
- Invalid or mismatched metadata must fail before the first `prepare_dense_sample.submit()` and must not replace an existing manifest; Task 2 retains both assertions.
- A failed submitted sample must stop further submission and leave the existing manifest unchanged; Task 2 retains the failure-boundary tests.
- GeoTIFF and Zarr publication must remain atomic and preserve an existing destination after a failed or competing write; Tasks 1 and 2 retain the storage failure tests.
- A generated workspace and built wheel must import only the layered paths, with no dependency on deleted domain packages; Task 3 adds stale-import and clean-artifact checks.

---

### Task 1: Restore configs, ingest flow, and stack task

**Files:**
- Create: `src/geosave_engine/workflow/configs/__init__.py`
- Create: `src/geosave_engine/workflow/configs/base.py`
- Create: `src/geosave_engine/workflow/configs/anchor.py`
- Create: `src/geosave_engine/workflow/flows/__init__.py`
- Create: `src/geosave_engine/workflow/flows/ingest.py`
- Create: `src/geosave_engine/workflow/tasks/__init__.py`
- Create: `src/geosave_engine/workflow/tasks/stack.py`
- Modify: `src/geosave_engine/cli/commands/workflow/ingest.py`
- Delete: `src/geosave_engine/workflow/ingestion/__init__.py`
- Delete: `src/geosave_engine/workflow/ingestion/anchor.py`
- Delete: `src/geosave_engine/workflow/ingestion/flow.py`
- Move: `tests/workflow/ingestion/test_anchor.py` to `tests/workflow/configs/test_anchor.py`
- Move: `tests/workflow/ingestion/test_flow.py` to `tests/workflow/flows/test_ingest.py`
- Move: `tests/workflow/ingestion/test_storage.py` to `tests/workflow/tasks/test_stack.py`
- Modify: `tests/cli/commands/test_workflow.py`

**Interfaces:**
- Consumes: `ModelSpec.load(path) -> ModelSpec`, `ModelSpec.load_rasters(anchor) -> dict[str, xr.Dataset]`, and native geodata stack persistence.
- Produces: `ConfigModel`; `AnchorConfig`; concrete anchor config models; `write_stack(rasters: Mapping[str, xr.Dataset], output: str | Path) -> str`; and Prefect flow `ingest(anchor: dict[str, JsonValue], *, output: str, spec: str) -> str`.

- [ ] **Step 1: Move the ingest tests to the layered paths and make the expected imports explicit**

Use `git mv` for the three test files. Update imports so config tests import from `workflow.configs`, flow tests import `ingest` from `workflow.flows` and its module from `workflow.flows.ingest`, and stack tests import public `write_stack` from `workflow.tasks.stack`. Rename test references from `_write_stack` to `write_stack`.

In `tests/cli/commands/test_workflow.py`, import `flows` plus `AnchorConfig` from `workflow.configs`, monkeypatch `flows.ingest`, and retain the assertions that malformed anchors never invoke the flow.

- [ ] **Step 2: Run the moved tests and verify the new packages are missing**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/configs tests/workflow/flows/test_ingest.py tests/workflow/tasks/test_stack.py tests/cli/commands/test_workflow.py -k ingest`

Expected: collection fails because `geosave_engine.workflow.configs`, `.flows`, or `.tasks` does not exist.

- [ ] **Step 3: Implement the layered ingest slice**

Move the strict `ConfigModel` policy to `configs/base.py`; move the anchor union and concrete configs to `configs/anchor.py`; export the five supported config names from `configs.__init__`.

Move `_write_stack` to `tasks/stack.py` as the public signature from the Interfaces block without changing its atomic local-Zarr contract. Keep `tasks.__init__.__all__` empty for this intermediate task because `write_stack` is an advanced operation imported from its owning module.

Move the Prefect flow to `flows/ingest.py`, import `AnchorConfig` from `workflow.configs` and `write_stack` from `workflow.tasks.stack`, and export `ingest` from `flows.__init__`. Update the CLI to call `workflow.flows.ingest` and parse with the layered `AnchorConfig`. Delete `workflow.ingestion` without an alias.

- [ ] **Step 4: Run the focused ingest slice**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/configs tests/workflow/flows/test_ingest.py tests/workflow/tasks/test_stack.py tests/cli/commands/test_workflow.py -k ingest`

Expected: PASS, including no-I/O missing-recipe validation and all stack atomicity tests.

- [ ] **Step 5: Lint and commit the ingest slice**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow tests/workflow tests/cli/commands/test_workflow.py && git diff --check`

Expected: PASS.

```bash
git add src/geosave_engine/workflow/configs src/geosave_engine/workflow/flows src/geosave_engine/workflow/tasks src/geosave_engine/workflow/ingestion src/geosave_engine/cli/commands/workflow/ingest.py tests/workflow/configs tests/workflow/flows tests/workflow/tasks tests/workflow/ingestion tests/cli/commands/test_workflow.py
git commit -m "refactor: restore layered ingest workflow"
```

### Task 2: Restore dense flow and preparation tasks

**Files:**
- Create: `src/geosave_engine/workflow/flows/prepare_dense_data.py`
- Create: `src/geosave_engine/workflow/tasks/dense.py`
- Create: `src/geosave_engine/workflow/tasks/manifest.py`
- Create: `src/geosave_engine/workflow/tasks/sample.py`
- Modify: `src/geosave_engine/workflow/flows/__init__.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`
- Modify: `src/geosave_engine/cli/commands/workflow/prepare_dense_data.py`
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Delete: `src/geosave_engine/workflow/training_data/__init__.py`
- Delete: `src/geosave_engine/workflow/training_data/dense.py`
- Delete: `src/geosave_engine/workflow/training_data/manifest.py`
- Delete: `src/geosave_engine/workflow/training_data/sample.py`
- Move: `tests/workflow/training_data/test_dense.py` to `tests/workflow/flows/test_prepare_dense_data.py`
- Move: `tests/workflow/training_data/test_sample_preparation.py` to `tests/workflow/tasks/test_dense.py`
- Move: `tests/workflow/training_data/test_manifest.py` to `tests/workflow/tasks/test_manifest.py`
- Move: `tests/workflow/training_data/test_metadata.py` to `tests/workflow/tasks/test_metadata.py`
- Move: `tests/workflow/training_data/test_sample.py` to `tests/workflow/tasks/test_sample.py`
- Modify: `tests/cli/commands/test_workflow.py`
- Modify: `tests/workflow/test_ingest_script.py`
- Modify: `tests/cli/core/test_workspace.py`

**Interfaces:**
- Consumes: Task 1's `workflow.flows`, `workflow.tasks`, and `write_stack`; `ModelSpec.load_rasters(anchor)`; current native sample and GeoParquet behavior.
- Produces: Prefect task `prepare_dense_sample(label: str | Path, model: ModelSpec, output: str | Path, *, format: SampleFormat = "geotiff", write_options: Mapping[str, JsonValue] | None = None) -> str`; Prefect flow `prepare_dense_data(labels: str, *, output: str, spec: str, pattern: str = "**/*.tif", max_concurrency: PositiveInt = 1, format: Literal["geotiff", "zarr"] = "geotiff", write_options: dict[str, JsonValue] | None = None, metadata: str | None = None) -> str`; and the existing public functions in the owning `tasks.sample` and `tasks.manifest` modules.

- [ ] **Step 1: Move dense tests and update them to specify the layered public interface**

Use `git mv` for all five test files. Update module imports to:

```python
from geosave_engine.workflow import flows, tasks
from geosave_engine.workflow.flows import prepare_dense_data
from geosave_engine.workflow.tasks import prepare_dense_sample
from geosave_engine.workflow.tasks.manifest import ...
from geosave_engine.workflow.tasks.sample import ...
```

Replace `test_training_data_exports_only_public_operations` with
`test_layers_export_only_supported_operations`. Assert
`flows.__all__ == ["ingest", "prepare_dense_data"]`,
`tasks.__all__ == ["prepare_dense_sample"]`, `flows.ingest` and
`flows.prepare_dense_data` are Prefect `Flow` objects,
`tasks.prepare_dense_sample` is a Prefect `Task`, and storage/manifest helpers
are not re-exported from `workflow.tasks`.

Update CLI tests to monkeypatch `workflow.flows.prepare_dense_data`. Update the
generated-script test and workspace-generation assertion to require
`from geosave_engine.workflow.flows import prepare_dense_data`.

- [ ] **Step 2: Run the dense suite and verify imports fail before implementation**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/flows/test_prepare_dense_data.py tests/workflow/tasks/test_dense.py tests/workflow/tasks/test_manifest.py tests/workflow/tasks/test_metadata.py tests/workflow/tasks/test_sample.py tests/cli/commands/test_workflow.py tests/workflow/test_ingest_script.py tests/cli/core/test_workspace.py`

Expected: collection fails because the layered dense flow and task modules do not yet exist.

- [ ] **Step 3: Move dense sample storage and manifest behavior without changing contracts**

Move `training_data/sample.py` to `tasks/sample.py` and
`training_data/manifest.py` to `tasks/manifest.py`; update the latter's relative
import. Preserve `SampleFormat`, GeoTIFF/Zarr validation, caller metadata
columns, relative paths, ordered manifest schema, laziness, and atomic writes.

- [ ] **Step 4: Split the Prefect task from the dense flow**

Create `tasks/dense.py` from `_validate_dense_sample` and
`prepare_dense_sample`; import storage from `tasks.sample`. Export only
`prepare_dense_sample` from `tasks.__init__`.

Create `flows/prepare_dense_data.py` from the current flow and bounded
submission loop; import discovery, metadata, paths, and manifest publication
from `tasks.manifest`, and import `prepare_dense_sample` from `workflow.tasks`.
Export `prepare_dense_data` beside `ingest` from `flows.__init__`.

Delete `workflow.training_data` rather than leaving import aliases.

- [ ] **Step 5: Update CLI and template consumers**

Make the CLI call `workflow.flows.prepare_dense_data`. Make the generated
`ingest_imagery.py` import the flow from `geosave_engine.workflow.flows`.
Do not change command names, parameters, defaults, or printed output.

- [ ] **Step 6: Run the focused dense and consumer suite**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/flows/test_prepare_dense_data.py tests/workflow/tasks/test_dense.py tests/workflow/tasks/test_manifest.py tests/workflow/tasks/test_metadata.py tests/workflow/tasks/test_sample.py tests/cli/commands/test_workflow.py tests/workflow/test_ingest_script.py tests/cli/core/test_workspace.py`

Expected: PASS, including bounded concurrency, failure stopping, metadata-before-submit, unchanged-manifest failure, resume validation, and both persistence formats.

- [ ] **Step 7: Run the complete workflow test tree and commit**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow tests/cli/commands/test_workflow.py tests/cli/core/test_workspace.py`

Expected: PASS.

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow src/geosave_engine/templates tests/workflow tests/cli && git diff --check`

Expected: PASS.

```bash
git add src/geosave_engine/workflow/flows src/geosave_engine/workflow/tasks src/geosave_engine/workflow/training_data src/geosave_engine/cli/commands/workflow/prepare_dense_data.py src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py tests/workflow/flows tests/workflow/tasks tests/workflow/training_data tests/cli/commands/test_workflow.py tests/cli/core/test_workspace.py tests/workflow/test_ingest_script.py
git commit -m "refactor: restore layered dense preparation"
```

### Task 3: Align documentation and prove the clean package

**Files:**
- Modify: `src/geosave_engine/workflow/__init__.py`
- Modify: `docs/guides/workflows.md`
- Modify: active package-path references found under `README.md`, `src`, and `tests`
- Test: package build and clean-export imports

**Interfaces:**
- Consumes: Task 1 and Task 2's final layered paths.
- Produces: active documentation and packaged source containing only `workflow.configs`, `workflow.flows`, and `workflow.tasks`.

- [ ] **Step 1: Add a stale-import guard to the workflow public-interface test**

In `tests/workflow/flows/test_prepare_dense_data.py`, add
`test_active_files_do_not_reference_removed_workflow_packages`. Scan Python
files under `src/geosave_engine` and `tests`, plus Markdown files under
`docs/guides` and the root `README.md`, excluding `__pycache__`, and assert
neither `geosave_engine.workflow.ingestion` nor
`geosave_engine.workflow.training_data` occurs. Keep historical
`docs/superpowers` records outside this assertion.

- [ ] **Step 2: Run the guard before documentation/source cleanup**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/flows/test_prepare_dense_data.py::test_active_files_do_not_reference_removed_workflow_packages`

Expected: FAIL if any active source or template still names a deleted package.

- [ ] **Step 3: Update active documentation and package description**

Describe `workflow` as serializable configs, reusable tasks, and deployable
Prefect flows. Update guide examples to import from `workflow.flows`; document
`workflow.configs` for anchor types and `workflow.tasks` only where direct task
composition is relevant. Do not rewrite historical specs or plans that record
the superseded migration.

- [ ] **Step 4: Run stale-path, lint, and scoped type gates**

Run: `rg -n "geosave_engine\.workflow\.(ingestion|training_data)|from geosave_engine\.workflow import (ingestion|training_data)" README.md docs/guides src tests -g '*.py' -g '*.md'`

Expected: no matches.

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src tests && UV_CACHE_DIR=/tmp/geosave-uv-cache uv run basedpyright src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow`

Expected: Ruff passes and BasedPyright reports 0 errors.

- [ ] **Step 5: Run complete verification**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q`

Expected: the complete non-slow suite passes with 10 tests deselected by current configuration.

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv build`

Expected: `dist/geosave_engine-0.2.0.tar.gz` and `dist/geosave_engine-0.2.0-py3-none-any.whl` build successfully. If sandbox DNS prevents resolving Hatchling, rerun the same command with network permission; do not change dependencies.

Export `HEAD` with `git archive` to a new temporary directory and run:

```bash
PYTHONPATH=<clean-export>/src .venv/bin/python -c "from geosave_engine.workflow.configs import AnchorConfig; from geosave_engine.workflow.flows import ingest, prepare_dense_data; from geosave_engine.workflow.tasks import prepare_dense_sample; print('workflow imports ok')"
```

Expected: `workflow imports ok`.

Inspect the wheel member list and assert it contains the three layered workflow
packages and no `workflow/ingestion/` or `workflow/training_data/` members.

- [ ] **Step 6: Commit documentation and verification guard**

```bash
git add src/geosave_engine/workflow/__init__.py docs/guides/workflows.md README.md tests/workflow/flows/test_prepare_dense_data.py
git commit -m "docs: align layered workflow API"
```
