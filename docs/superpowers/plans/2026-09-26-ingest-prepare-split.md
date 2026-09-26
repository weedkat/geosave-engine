# Ingest and Training Preparation Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give single-anchor acquisition and label-driven training preparation separate public Prefect flows.

**Architecture:** Restore `ingest` as one anchor to one Zarr. Move label discovery, sample fan-out, resume, and GeoVector publication to `prepare_training`; both flows share native loader and writer functions rather than nesting flows.

**Tech Stack:** Python 3.12, Prefect, Pydantic, xarray/DataTree, GeoVector/GeoParquet, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-ingest-prepare-split-design.md`

## Global Constraints

- Keep `model_spec.yaml` model-owned; paths, globs, outputs, and runtime source settings remain caller-owned.
- Keep native xarray objects task-local and pass completed paths across Prefect seams.
- Preserve lazy reads until atomic persistence.
- Use native library failures except for the domain invariants named in the spec.
- Preserve unrelated worktree changes.

## Review Focus

- The single-anchor flow must not import training preparation code.
- The training flow must derive anchors from labels and never parse `AnchorConfig`.
- Relative output paths must reopen correctly through the manifest.
- Mixed native CRSs must coexist in one catalog without losing grid metadata.
- A failed training run must not replace the existing manifest.

---

### Task 1: Restore Focused Single-Anchor Ingestion

**Files:**
- Create: `src/geosave_engine/workflow/configs/anchor.py`
- Modify: `src/geosave_engine/workflow/configs/__init__.py`
- Modify: `src/geosave_engine/workflow/flows/ingest.py`
- Create: `tests/workflow/configs/test_anchor.py`
- Rewrite: `tests/workflow/flows/test_ingest.py`

**Interfaces:**
- Produces: `ingest(sources, anchor, *, output, spec) -> str` and the Pydantic `AnchorConfig` union.
- Consumes: `RasterLoader.load(SourceConfig, GeoAnchor)` and `write_stack(...)`.

- [ ] Write a failing real test proving one raster anchor produces one source-only Zarr.
- [ ] Run the test and confirm the current bulk signature fails.
- [ ] Restore concise anchor configs and implement the one-anchor flow with `TypeAdapter(AnchorConfig).validate_python`.
- [ ] Test exact source bindings and native anchor opening; remove bulk assertions from this test module.
- [ ] Run `tests/workflow/configs/test_anchor.py` and `tests/workflow/flows/test_ingest.py`.
- [ ] Commit as `refactor: restore focused raster ingestion`.

### Task 2: Move Training Preparation to Its Own Flow

**Files:**
- Create: `src/geosave_engine/workflow/flows/prepare_training.py`
- Modify: `src/geosave_engine/workflow/flows/__init__.py`
- Rename: `src/geosave_engine/workflow/tasks/ingest.py` to `prepare.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`
- Create: `tests/workflow/flows/test_prepare_training.py`
- Rename: `tests/workflow/tasks/test_ingest.py` to `test_prepare.py`
- Modify: `src/geosave_engine/workflow/tasks/catalog.py`
- Modify: `tests/workflow/tasks/test_catalog.py`

**Interfaces:**
- Produces: `prepare_training(labels, sources, *, output, spec, pattern="**/*.tif") -> str` and `prepare_sample(...) -> str`.
- Consumes: native loader/writer functions and `save_catalog(...)`.

- [ ] Write a failing two-label flow test using the new name and assert mirrored Zarrs plus an ordered manifest.
- [ ] Move the bulk orchestration and rename the sample task.
- [ ] Simplify discovery to sorted files plus only empty-result and output-collision checks.
- [ ] Retain resume, partial-failure, relative-path, and mixed-CRS behavioral tests; delete defensive path-policy tests.
- [ ] Run all workflow task and flow tests, including the slow preparation tests.
- [ ] Commit as `refactor: separate training data preparation`.

### Task 3: Point the Workspace at Training Preparation

**Files:**
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Modify: `tests/workflow/test_ingest_script.py`

**Interfaces:**
- Consumes: public `prepare_training`.
- Produces: the same thin Typer CLI, printing the GeoParquet manifest path.

- [ ] Change the script test to require `prepare_training` and confirm it fails against the current import.
- [ ] Update the script and keep environment/GDAL setup unchanged.
- [ ] Run workflow tests, Ruff, BasedPyright, and `git diff --check`.
- [ ] Run the full suite and report unrelated ML collection failures exactly.
- [ ] Commit as `refactor: prepare training data from workspace script`.
