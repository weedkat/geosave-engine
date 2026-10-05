# Workflows on STAC Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run dense preparation and ingestion on the library's own readers, writers, and STAC tables: labels come in as a table, samples are registered from the files written, and the manifest is recorded as samples finish.

**Architecture:** `read_stack` learns to open a directory of one raster per layer, which closes the gap that `open_sample` and `sample_assets` filled. A sample row is `GeoVector.from_xarray(read_stack(folder))`. The dense flow upserts each returned row into the manifest and resumes from it. `ingest` writes the same sample layout and can record its row into a catalog.

**Tech Stack:** Python 3.12, Prefect, GeoPandas, xarray, Typer, pytest

**Spec:** `docs/superpowers/specs/2026-10-04-workflow-on-stac-tables-design.md`

## Global Constraints

- `write_sample` keeps its current per-layer writers and its atomic staging; only `open_sample` and `sample_assets` leave `sample.py`.
- A sample row is built only with `GeoVector.from_xarray(read_stack(folder), ...)`.
- A label table's label asset is named `label`. Model raster name `label` stays reserved.
- New names use `validate`, never `check`. No participle names for locals.
- No compatibility alias for `--metadata`, `write_manifest`, `read_sample_metadata`, `write_stack`, `open_sample`, or `sample_assets`.
- Prefect validates typed flow parameters; do not re-validate them by hand.
- Do not commit, stage, stash, checkout, or restore anything. Do not touch `.ipynb` files.

## Review Focus

- A sample directory left by a crash, with no manifest row, must be registered and not rewritten or re-downloaded.
- A manifest written by an earlier run with a different spec must stop the run, not be silently reused.
- A staging directory (`.name-xxxx`) left beside samples must not be read as a layer or a sample.
- Caller columns holding nulls or lists must survive from the label table to the manifest.
- `ingest --catalog` on a catalog that already holds other scenes must keep them.

## File Structure

| File | Change |
| --- | --- |
| `src/geosave_engine/geodata/utils/io/__init__.py` | `read_stack` opens a directory |
| `src/geosave_engine/workflow/tasks/sample.py` | `open_sample`, `sample_assets` removed |
| `src/geosave_engine/workflow/tasks/labels.py` | New: `find_labels`, `read_labels` |
| `src/geosave_engine/workflow/tasks/dense.py` | `prepare_dense_sample` returns its row; `validate_row` |
| `src/geosave_engine/workflow/tasks/manifest.py`, `stack.py` | Deleted |
| `src/geosave_engine/workflow/flows/prepare_dense_data.py` | Label table, recorder, resume |
| `src/geosave_engine/workflow/flows/ingest.py` | Sample directory, `format`, `catalog` |
| `src/geosave_engine/cli/commands/workflow/*.py` | Options |
| `src/geosave_engine/templates/boilerplate/scripts/` | Follow the flow's parameters |
| `tests/geodata/utils/io/test_read_stack.py` | New |
| `tests/workflow/**` | Rewritten around the new behaviour |
| `docs/guides/workflows.md` | Updated |

---

### Task 1: `read_stack(directory)`

**Interfaces:** Produces `read_stack(source)` accepting a directory: one group per entry named by its stem; hidden entries and non-raster files skipped; groups ordered by name; `ValueError` when it holds no raster.

- [ ] Write `tests/geodata/utils/io/test_read_stack.py`: a `stack.gs.to_cog` directory; a directory of `.zarr` layers; a mixed one; a timed group written as a tree; hidden staging directory and a `manifest.parquet` beside the layers are skipped; empty directory raises; laziness (no dask task starts); `GeoVector.from_xarray(read_stack(dir))` keys assets by layer.
- [ ] Run and watch them fail on the suffix error.
- [ ] Implement in `read_stack`, before the suffix dispatch, for a directory whose suffix is not `.zarr` or a product suffix.
- [ ] Run `uv run pytest tests/geodata/utils/io -q`.

### Task 2: Samples register through the library

**Interfaces:** Consumes Task 1. Produces `prepare_dense_sample(label_path, spec, output, *, sample_id, properties=None, format="geotiff", write_options=None) -> GeoDataFrame` and `validate_row(row, spec) -> None` in `tasks/dense.py`; `find_labels(root, pattern) -> dict[str, Path]` and `read_labels(source, pattern) -> GeoDataFrame` in `tasks/labels.py`.

- [ ] Rewrite `tests/workflow/tasks/test_dense.py`, `test_sample.py`, and a new `test_labels.py`:
  - the task returns one row with `id`, assets `label` plus the spec's rasters, the label's caller columns, and time;
  - an existing sample directory is registered without STAC access and without rewriting;
  - `validate_row` refuses a row whose assets are not exactly `label` plus the spec's rasters;
  - a label without time raises `Label raster has no time`;
  - `read_labels` on a directory and on the equivalent table give the same ids, label hrefs, and time; a table lacking `id`, `assets`, or a `label` asset raises; caller columns are kept;
  - `test_sample.py` drops the `open_sample` and `sample_assets` tests and reads samples back with `read_stack`.
- [ ] Run and watch them fail.
- [ ] Implement `labels.py`, the new `dense.py`, trim `sample.py`; delete `manifest.py`, `stack.py`, `tests/workflow/tasks/test_manifest.py`, `test_stack.py`.
- [ ] Run `uv run pytest tests/workflow/tasks -q`.

### Task 3: `prepare-dense-data` records as it goes

**Interfaces:** Consumes Task 2. Produces `prepare_dense_data(labels, *, output, spec, pattern="**/*.tif", max_concurrency=1, format="geotiff", write_options=None) -> str`.

- [ ] Rewrite `tests/workflow/flows/test_prepare_dense_data.py`:
  - a label directory and the equivalent label table produce the same manifest;
  - caller columns of the label table reach the manifest, nulls included;
  - when a later sample fails, the manifest holds the finished ones and a rerun prepares only the rest;
  - a recorded row is reused without calling the task; a row whose assets do not match the spec stops the run;
  - a sample directory without a row is registered, not prepared again;
  - a file at the manifest path that is not a STAC table stops the run before submission;
  - the finished manifest is in label order and holds exactly the label ids;
  - concurrency bound and stop-after-failure tests keep their intent.
- [ ] Run and watch them fail.
- [ ] Implement the flow.
- [ ] Run `uv run pytest tests/workflow -q`.

### Task 4: `ingest` writes a sample directory and records it

**Interfaces:** Produces `ingest(anchor, *, output, spec, format="zarr", write_options=None, catalog=None) -> str`.

- [ ] Rewrite `tests/workflow/flows/test_ingest.py`: the output is a directory `read_stack` reopens; with `catalog`, one row is added to a new table and to an existing one without losing its rows, and a row carrying the same id is replaced; an existing output directory is refused.
- [ ] Run and watch them fail.
- [ ] Implement.
- [ ] Run `uv run pytest tests/workflow/flows/test_ingest.py -q`.

### Task 5: CLI, template script, guide

- [ ] Update `tests/cli/commands/test_workflow.py` and `tests/workflow/test_prepare_dense_data_script.py`: `--metadata` is gone, `ingest` forwards `--format` and `--catalog`.
- [ ] Update both CLI commands, the boilerplate script, and `docs/guides/workflows.md`.
- [ ] Final verification: `uv run pytest -q`; `uv run basedpyright` on the changed files; `uv run ruff check src tests`; `git diff --check`; `grep -rn "open_sample\|sample_assets\|write_manifest\|read_sample_metadata\|write_stack\|--metadata" src tests docs/guides` returns nothing.
- [ ] Report what changed, checks run, breaking changes. Do not commit.
