# Dense Manifest Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish minimal portable dense manifests and merge caller-owned per-label columns from CSV, TSV, Parquet, or XLSX tables.

**Architecture:** A focused workflow metadata module reads and validates the optional sidecar before Prefect submits any sample task. Dense preparation keeps its internal relative label keys for output placement, while manifest publication writes only the prepared path, format, time, CRS, dimensions, custom columns, and geometry.

**Tech Stack:** Python 3.12, pandas, OpenPyXL, PyArrow, GeoPandas/GeoParquet, Prefect, Typer, pytest

**Spec:** `docs/superpowers/specs/2026-09-29-dense-manifest-metadata-design.md`

## Global Constraints

- Support exactly `.csv`, `.tsv`, `.parquet`, and `.xlsx` metadata files.
- Resolve `label_path` relative to the metadata table's parent directory; require relative paths but allow `..`.
- Validate an exact one-to-one match with labels discovered under `--labels` before submitting sample tasks.
- Do not add a manifest model, metadata registry, remote enrichment, or automatic country/biome lookup.
- Omit `label_path`, `sample_id`, `variables`, and `grid_transform` from dense manifests.
- Preserve generic `GeoVector.from_xarray` variable and grid-field behavior.
- Preserve bounded ingestion, resumable sample validation, lazy pixel behavior, and atomic manifest replacement.
- Keep unrelated dirty-worktree changes intact.

## Review Focus

- Two different relative spellings resolving to one label must be rejected as duplicate `label_path` values; Task 1 tests this.
- An uppercase supported suffix such as `.CSV` should use the same reader as lowercase; Task 1 tests suffix normalization.
- XLSX workbooks with multiple worksheets must read only the first worksheet; Task 1 tests this explicitly.
- A metadata file that exists but has no rows must report the discovered labels as missing before task submission; Tasks 1 and 3 test this path.
- Custom columns containing null values must survive both record construction and GeoParquet round-trip; Task 2 tests this.

---

### Task 1: Read and validate per-label metadata tables

**Files:**
- Create: `src/geosave_engine/workflow/metadata.py`
- Create: `tests/workflow/test_metadata.py`

**Interfaces:**
- Consumes: discovered labels as `dict[str, Path]`, where each key is the existing suffix-free relative output key.
- Produces: `read_sample_metadata(source: str | Path | None, labels: dict[str, Path]) -> dict[str, dict[str, object]]`, keyed by the same internal label keys and preserving caller column order.

- [ ] **Step 1: Write failing reader tests**

Add parameterized tests proving CSV, TSV, Parquet, and XLSX return the same mapping for a table whose `label_path` values resolve from the table parent. Add an XLSX workbook with a conflicting second sheet and assert only the first sheet is read. Add an uppercase `.CSV` case.

- [ ] **Step 2: Run the reader tests and verify red**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/test_metadata.py -q`

Expected: collection/import failure because `geosave_engine.workflow.metadata` does not exist.

- [ ] **Step 3: Implement format dispatch**

Implement `read_sample_metadata(source, labels)` and private `_read_table(source: Path) -> pd.DataFrame`. Dispatch case-insensitive suffixes to `pd.read_csv`, `pd.read_csv(sep="\t")`, `pd.read_parquet`, or `pd.read_excel(sheet_name=0, engine="openpyxl")`; reject every other suffix with the supported suffixes in the error.

- [ ] **Step 4: Write failing validation tests**

Test missing `label_path`; null, non-string, and absolute values; duplicate source rows; two spellings resolving to one path; missing and extra resolved paths; non-string columns; and collisions with `path`, `format`, time, grid, and `geometry`. Assert every case fails with the offending field or path in the message. Assert `source=None` returns empty properties for every label.

- [ ] **Step 5: Implement exact path validation**

Resolve each relative `label_path` from `source.parent`, match resolved paths against `labels.values()`, and return caller columns excluding `label_path`. Require an exact set match and preserve the input table's row and column values without inventing a schema.

- [ ] **Step 6: Run Task 1 tests and verify green**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/test_metadata.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/geosave_engine/workflow/metadata.py tests/workflow/test_metadata.py
git commit -m "feat: read dense sample metadata tables"
```

### Task 2: Publish the minimal manifest schema

**Files:**
- Modify: `src/geosave_engine/workflow/tasks/catalog.py`
- Modify: `tests/workflow/tasks/test_catalog.py`

**Interfaces:**
- Consumes: `write_manifest(samples, destination, *, format, metadata)` where `metadata` is the mapping produced by Task 1.
- Produces: an atomic GeoParquet manifest ordered as `path`, `format`, time fields, CRS/dimensions, caller columns, then `geometry`.

- [ ] **Step 1: Replace the current schema assertion with failing minimal-schema tests**

Assert raw GeoParquet has no `sample_id`, `variables`, `grid_transform`, or `label_path`; has `path`, `format`, `start_datetime`, `end_datetime`, `grid_crs`, `grid_height`, `grid_width`, and `geometry`; stores its sample path relatively; and resolves that path through `io.read_vector`.

- [ ] **Step 2: Add failing custom-column tests**

Pass ordered custom properties for each internal sample key and assert string, numeric, boolean, and null values round-trip in caller column order. Keep the existing Dask callback assertion proving no sample pixels compute.

- [ ] **Step 3: Run catalog tests and verify red**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_catalog.py -q`

Expected: failures because the old writer still emits `sample_id`, `variables`, and `grid_transform` and accepts no metadata mapping.

- [ ] **Step 4: Implement minimal record construction**

Extend `write_manifest` with `metadata: Mapping[str, Mapping[str, object]] | None = None`. Build each record with `GeoVector.from_xarray(..., fields=("time",), path=..., format=..., grid_crs=..., grid_height=..., grid_width=..., **caller_properties)`, then deterministically reorder the concatenated frame before the existing atomic `to_geoparquet` call.

- [ ] **Step 5: Run catalog tests and verify green**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/tasks/test_catalog.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/geosave_engine/workflow/tasks/catalog.py tests/workflow/tasks/test_catalog.py
git commit -m "refactor: minimize dense manifest schema"
```

### Task 3: Connect metadata to the flow and CLI

**Files:**
- Modify: `src/geosave_engine/workflow/flows/prepare_dense_data.py`
- Modify: `src/geosave_engine/cli/commands/workflow/prepare_dense_data.py`
- Modify: `tests/workflow/flows/test_prepare_dense_data.py`
- Modify: `tests/cli/commands/test_workflow.py`
- Modify: `docs/guides/workflows.md`

**Interfaces:**
- Consumes: Task 1 `read_sample_metadata`; Task 2 `write_manifest(..., metadata=...)`.
- Produces: optional `metadata: str | None = None` on the flow and optional `--metadata PATH` on the CLI.

- [ ] **Step 1: Write failing flow tests**

Add a sidecar outside the labels directory whose relative paths traverse into the labels directory. Assert caller columns appear in the finished manifest, duplicate filename stems in separate subdirectories join correctly, `label_path` is absent, and a metadata-only rerun performs no new STAC requests. Update existing flow assertions to use manifest `path` rather than `sample_id`.

- [ ] **Step 2: Write failing pre-submission tests**

Parameterize invalid and empty sidecars and monkeypatch `prepare_dense_sample.submit` to fail if called. Assert metadata validation raises first and an existing manifest remains byte-for-byte unchanged.

- [ ] **Step 3: Run flow tests and verify red**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/flows/test_prepare_dense_data.py -q`

Expected: failures because the flow has no `metadata` parameter or validation call.

- [ ] **Step 4: Implement flow integration**

Add `metadata: str | None = None`, call `read_sample_metadata(metadata, discovered)` immediately after discovery and before submission, and pass the returned mapping to `write_manifest`. Update the Google-style docstring.

- [ ] **Step 5: Write and implement the CLI forwarding test**

Extend the existing Typer test with `--metadata data/sample-metadata.xlsx`, assert the flow receives its string path, add the optional `Path` parameter, and update CLI help text.

- [ ] **Step 6: Update workflow documentation**

Document the four supported formats, table-relative `label_path`, exact matching, custom-column preservation, and the minimal output schema. Do not document automatic country or biome enrichment.

- [ ] **Step 7: Run Task 3 tests and verify green**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/geosave_engine/workflow/flows/prepare_dense_data.py src/geosave_engine/cli/commands/workflow/prepare_dense_data.py tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py docs/guides/workflows.md
git commit -m "feat: merge custom dense manifest metadata"
```

### Task 4: Verify the complete change

**Files:**
- Modify only if verification reveals a task-local defect.

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: verified implementation matching the design.

- [ ] **Step 1: Run focused workflow tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/test_metadata.py tests/workflow/tasks/test_catalog.py tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run scoped lint**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow tests/workflow tests/cli/commands/test_workflow.py`

Expected: no diagnostics.

- [ ] **Step 3: Run the full suite**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q`

Expected: all tests pass, apart from explicitly deselected markers already configured by the repository.

- [ ] **Step 4: Run diff validation**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Smoke-test the real CLI**

Create a temporary CSV beside a relative label reference and run one existing example label through `uv run geosave workflow prepare-dense-data --metadata ...`. Inspect raw GeoParquet with GeoPandas and assert the custom column is present, the stored sample path is relative, and `label_path`, `sample_id`, `variables`, and `grid_transform` are absent.

- [ ] **Step 6: Request code review**

Use `superpowers:requesting-code-review` for the complete diff and resolve any verified findings before handoff.
