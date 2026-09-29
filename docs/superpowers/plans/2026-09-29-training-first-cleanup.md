# Training-first Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stale prediction and technical-layer workflow code with a tensor-clean Lightning task, a top-level behavior-rich model-spec package, domain-owned ingestion/training-data workflows, and a release package.

**Architecture:** `geosave_engine.model_spec` owns the model YAML contract and its acquisition/preprocessing behavior. `workflow.ingestion` and `workflow.training_data` own durable jobs and artifacts without shared technical `flows`, `tasks`, or `configs` buckets; `release` owns versioned artifacts and its Hugging Face adapter.

**Tech Stack:** Python 3.12, Pydantic v2, xarray/Dask, PyTorch Lightning, Prefect 3, PySTAC Client, GeoPandas/Parquet, Hugging Face Hub, Transformers, pytest, Ruff, BasedPyright.

**Spec:** `docs/superpowers/specs/2026-09-29-training-first-cleanup-design.md`

## Global Constraints

- Preserve the user's unrelated dirty-worktree changes; stage only files owned by each task.
- Do not add compatibility aliases for removed Alpha paths.
- Keep `ModelSpec.load()` inert; imports, network access, and declared calls begin only at explicit runtime methods.
- A semantic-segmentation model returns exactly one logits tensor; keep Lightning `predict_step`.
- GeoSave must not implicitly compute raster pixels; explicit `chunks: null` and user calls may be eager.
- Only independently runnable jobs are Prefect flows; only per-sample bounded work is submitted as a task.
- Keep `max_concurrency=1` as the dense preparation default.
- Do not change geodata behavior or add prediction, few-shot, autoregressive, MLflow, or runner abstractions.
- Public interfaces receive concise Google-style docstrings.
- Source tests mirror the final source package layout.

## Review Focus

- A model with several rasters and one missing STAC recipe must list every missing name before opening any endpoint; Task 3 adds this test.
- A missing collection may fall back to the next STAC endpoint, while malformed collection metadata must retain its native error and stop; Task 3 adds both tests.
- Removed prediction keys in `model_spec.yaml` must fail strict validation instead of being ignored; Task 2 adds this test.
- Invalid or mismatched metadata must fail before `prepare_dense_sample.submit()` and must not replace an existing manifest; Task 5 adds this test.
- Importing `geosave_engine.release` and loading only a spec must work without importing optional Transformers; Task 6 adds a fresh-process test.

---

### Task 1: Finish the Lightning package move and tensor contract

**Files:**
- Modify: `src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py`
- Modify: `src/geosave_engine/ml/lightning/__init__.py`
- Modify: `src/geosave_engine/templates/common/main.py`
- Modify: `tests/ml/lightning/tasks/test_semantic_segmentation.py`
- Modify: `tests/ml/test_cli.py`
- Modify: `tests/cli/core/test_workspace.py`
- Delete: `src/geosave_engine/ml/cli.py` if it still exists

**Interfaces:**
- Consumes: existing `ModelChain(**model_inputs)` behavior.
- Produces: `SemanticSegmentationTask.forward(**model_inputs: Any) -> torch.Tensor`, unchanged `predict_step(...) -> tuple[torch.Tensor, Any]`, and the sole CLI path `geosave_engine.ml.lightning.cli.GeosaveCLI`.

- [ ] **Step 1: Replace the mapped-result acceptance test with a tensor-contract rejection test**

Assert that a chain returning `{"logits": tensor}` raises `TypeError` containing `must return logits as a tensor`; retain the direct `predict_step` and Lightning tiled-prediction assertions.

- [ ] **Step 2: Run the focused task test and verify it fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/ml/lightning/tasks/test_semantic_segmentation.py -k 'mapping or predict'`

Expected: the mapping case does not yet raise because `_logits()` accepts it.

- [ ] **Step 3: Narrow `forward()` and remove `_logits()`**

Implement `forward(self, **model_inputs: Any) -> torch.Tensor`; validate the direct chain result once and use `self(**model_inputs)` directly in train, validation, test, and prediction steps. Remove the unused `Mapping` import and update `predict_step` documentation to describe raw logits without model-spec postprocessing.

- [ ] **Step 4: Move every stale `GeosaveCLI` import to the Lightning package**

Update source templates, test imports, monkeypatch strings, and workspace assertions. Export only intentionally supported Lightning names from `ml/lightning/__init__.py`; do not restore `ml.cli`.

- [ ] **Step 5: Run focused tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/ml/test_cli.py tests/ml/lightning/tasks/test_semantic_segmentation.py tests/cli/core/test_workspace.py`

Expected: PASS.

- [ ] **Step 6: Commit the slice**

```bash
git add src/geosave_engine/ml/lightning src/geosave_engine/templates/common/main.py tests/ml tests/cli/core/test_workspace.py
git commit -m "refactor: enforce lightning tensor outputs"
```

### Task 2: Remove unfinished prediction contracts

**Files:**
- Modify: `src/geosave_engine/workflow/specs/model.py`
- Modify: `src/geosave_engine/workflow/specs/__init__.py`
- Modify: `src/geosave_engine/release.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml`
- Modify: `tests/workflow/specs/test_model.py`
- Modify: `tests/workflow/specs/test_examples.py`
- Modify: `tests/test_release.py`
- Delete: `src/geosave_engine/workflow/prediction.py`
- Delete: `src/geosave_engine/workflow/flows/predict.py`
- Delete: `src/geosave_engine/workflow/tasks/predict.py`
- Delete: `src/geosave_engine/workflow/specs/prediction.py`
- Delete: `tests/workflow/flows/test_predict.py`
- Delete: `tests/workflow/tasks/test_predict.py`
- Delete: `tests/workflow/test_prediction.py`
- Delete: `docs/superpowers/specs/2026-09-27-dense-prediction-design.md`
- Delete: `docs/superpowers/plans/2026-09-27-dense-prediction.md`

**Interfaces:**
- Consumes: strict `SpecModel` validation and the release bundle contract.
- Produces: `ModelSpec(schema_version, rasters, preprocessing)` only; no production prediction interface.

- [ ] **Step 1: Add strict rejection and release tests**

Parameterize removed keys `tiling`, `model_inputs`, `aggregation`, `postprocessing`, and `exports`; assert each produces a Pydantic extra-field error. Update release fixtures to save a spec containing only rasters and preprocessing, and assert release validation no longer reads `model_inputs`.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/specs/test_model.py tests/workflow/specs/test_examples.py tests/test_release.py`

Expected: removed fields are still accepted or release validation expects `model_inputs`.

- [ ] **Step 3: Delete prediction implementation and fields**

Remove the five prediction-only `ModelSpec` fields, prediction exports, flow/task exports, implementation files, tests, and stale prediction documents. Remove model-input compatibility checking from `_validate_release`; retain serializable stage-spec and complete-file checks.

- [ ] **Step 4: Simplify the template model specification**

Keep only `schema_version`, `rasters`, and deterministic `preprocessing`; ensure the fixture example parses through strict `ModelSpec.load()`.

- [ ] **Step 5: Run the focused tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/specs tests/test_release.py`

Expected: PASS.

- [ ] **Step 6: Commit the slice**

```bash
git add src/geosave_engine/workflow src/geosave_engine/release.py src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml tests/workflow tests/test_release.py docs/superpowers
git commit -m "refactor: remove unfinished prediction workflow"
```

### Task 3: Establish the top-level model-spec package

**Files:**
- Move: `src/geosave_engine/workflow/specs/` to `src/geosave_engine/model_spec/`
- Modify: `src/geosave_engine/model_spec/model.py`
- Modify: `src/geosave_engine/model_spec/stage.py`
- Modify: `src/geosave_engine/model_spec/stac.py`
- Modify: `src/geosave_engine/model_spec/__init__.py`
- Move: `tests/workflow/specs/` to `tests/model_spec/`
- Modify: every source, test, and template import of `geosave_engine.workflow.specs`
- Delete: `src/geosave_engine/workflow/tasks/load.py`
- Delete: `src/geosave_engine/workflow/tasks/process.py`
- Delete: `tests/workflow/tasks/test_load.py`
- Delete: `tests/workflow/tasks/test_process.py`

**Interfaces:**
- Consumes: `GeoAnchor`, `StacClient`, `StacSource`, `RasterRequirement.select_raster()`, and `CallSpec.invoke()`.
- Produces: `StacRecipe.load_raster(anchor: GeoAnchor, /) -> xr.Dataset`, `StageSpec.run(inputs: Mapping[str, Any], /) -> dict[str, Any]`, `ModelSpec.load_rasters(anchor: GeoAnchor, /) -> dict[str, xr.Dataset]`, and `ModelSpec.preprocess(inputs: Mapping[str, xr.Dataset], /) -> dict[str, Any]`.

- [ ] **Step 1: Move model-spec tests and add behavioral cases**

Add tests for inert YAML loading, preprocessing-root rejection, ordered execution, rebinding, forward-reference failure before invocation, no caller-mapping mutation, raster selection, all-recipes preflight, declaration-order results, contextual error notes, default lazy chunks, and explicit eager chunks.
Move the `scale`, `value`, and `audit` runtime-root examples out of the
model-spec fixture into direct `StageSpec` tests; model fixtures may reference
only declared raster roots.

- [ ] **Step 2: Add endpoint distinction tests**

Assert that the exact missing-collection `ValueError` falls back in endpoint order, malformed collection data does not fall back, empty search does not fall back, and cached clients still create fresh sources.

- [ ] **Step 3: Run the moved focused tests and verify failures**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/model_spec`

Expected: imports or new methods fail before implementation.

- [ ] **Step 4: Implement behavior on its owning declarations**

Move the existing endpoint/cache/source logic into `StacRecipe.load_raster()`, the ordered state loop into `StageSpec.run()`, and model-level preflight/selection/composition into `ModelSpec.load_rasters()` and `ModelSpec.preprocess()`. Add private `ModelSpec._validated()`, make saving use it, and reject preprocessing roots absent from `rasters` without resolving calls.

- [ ] **Step 5: Update imports without aliases**

Import public declarations from `geosave_engine.model_spec`. Update workflows, release, templates, and tests in the same step; do not leave `workflow.specs` forwarding modules.

- [ ] **Step 6: Run model-spec and immediate-consumer tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/model_spec tests/workflow/flows/test_ingest.py tests/workflow/tasks/test_dense.py tests/test_release.py`

Expected: PASS.

- [ ] **Step 7: Commit the slice**

```bash
git add src/geosave_engine/model_spec src/geosave_engine/workflow src/geosave_engine/release.py src/geosave_engine/templates tests/model_spec tests/workflow tests/test_release.py
git commit -m "refactor: make model spec a top-level package"
```

### Task 4: Move ingestion to its domain package

**Files:**
- Create: `src/geosave_engine/workflow/ingestion/__init__.py`
- Create: `src/geosave_engine/workflow/ingestion/anchor.py`
- Create: `src/geosave_engine/workflow/ingestion/flow.py`
- Move: `tests/workflow/configs/test_anchor.py` to `tests/workflow/ingestion/test_anchor.py`
- Move: `tests/workflow/flows/test_ingest.py` to `tests/workflow/ingestion/test_flow.py`
- Modify: `src/geosave_engine/cli/commands/workflow/ingest.py`
- Modify: `tests/cli/commands/test_workflow.py`
- Delete: ingestion-owned files from `workflow/configs`, `workflow/flows`, and `workflow/tasks/save.py`

**Interfaces:**
- Consumes: `ModelSpec.load()`, `ModelSpec.load_rasters()`, native geodata stack/Zarr I/O.
- Produces: `AnchorConfig` and `ingest(anchor: dict[str, JsonValue], *, output: str, spec: str) -> str` from `geosave_engine.workflow.ingestion`.

- [ ] **Step 1: Move ingestion tests to the target package and update imports**

Keep assertions for coordinate, GeoJSON, and raster-backed anchors; atomic local `.zarr` publication; missing-recipe failure; no partial destination; and CLI argument forwarding.

- [ ] **Step 2: Run ingestion tests and verify import failures**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/ingestion tests/cli/commands/test_workflow.py -k ingest`

Expected: target package is absent.

- [ ] **Step 3: Implement the ingestion package**

Move `ConfigModel` and anchor unions into `ingestion/anchor.py`. Put the Prefect `ingest` flow and its private atomic `_write_stack(...) -> str` implementation in `ingestion/flow.py`; call `model.load_rasters()` rather than extracting requirements.

- [ ] **Step 4: Update CLI and package exports**

Import `AnchorConfig` and `ingest` from `workflow.ingestion`; delete the old flow/config exports without aliases.

- [ ] **Step 5: Run focused tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/ingestion tests/cli/commands/test_workflow.py -k ingest`

Expected: PASS.

- [ ] **Step 6: Commit the slice**

```bash
git add src/geosave_engine/workflow/ingestion src/geosave_engine/cli/commands/workflow/ingest.py src/geosave_engine/workflow tests/workflow/ingestion tests/cli/commands/test_workflow.py
git commit -m "refactor: group raster ingestion by domain"
```

### Task 5: Move dense preparation to the training-data package

**Files:**
- Create: `src/geosave_engine/workflow/training_data/__init__.py`
- Create: `src/geosave_engine/workflow/training_data/dense.py`
- Create: `src/geosave_engine/workflow/training_data/sample.py`
- Create: `src/geosave_engine/workflow/training_data/manifest.py`
- Move: dense flow/task/catalog/save/metadata tests to `tests/workflow/training_data/`
- Modify: `src/geosave_engine/cli/commands/workflow/prepare_dense_data.py`
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py`
- Modify: `tests/cli/commands/test_workflow.py`
- Modify: `tests/workflow/test_ingest_script.py`
- Delete: remaining old `workflow/flows`, `workflow/tasks`, `workflow/configs`, and root `workflow/metadata.py`

**Interfaces:**
- Consumes: complete `ModelSpec`, `ModelSpec.load_rasters()`, native raster I/O, Prefect futures, and `GeoVector`.
- Produces: `prepare_dense_data(...) -> str` and `open_sample(source: str | Path, *, format: SampleFormat) -> xr.DataTree` from `workflow.training_data`; `prepare_dense_sample` remains package implementation submitted only by the flow.

- [ ] **Step 1: Move tests and add metadata ordering/failure tests**

Test CSV/TSV/Parquet/XLSX first-sheet reading through manifest behavior, stable relative sample IDs, reserved-column rejection, exact label-set matching, deterministic custom-column ordering, metadata forwarded into rows, default concurrency of one, bounded submission, stop-after-first-failure, existing-sample resume, and unchanged manifest on failure.

- [ ] **Step 2: Add the pre-submission regression test**

Monkeypatch `prepare_dense_sample.submit`; supply invalid metadata and assert the submit spy is never called and an existing manifest remains byte-for-byte unchanged.

- [ ] **Step 3: Run target tests and verify import failures**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/training_data tests/cli/commands/test_workflow.py -k 'prepare_dense or metadata'`

Expected: target package is absent.

- [ ] **Step 4: Implement the sample artifact module**

Move `SampleFormat`, `open_sample`, `write_sample`, GeoTIFF normalization, and completed-sample validation into `training_data/sample.py`. Export `open_sample` and `SampleFormat`; keep publication/validation helpers internal to the package.

- [ ] **Step 5: Implement the manifest module**

Co-locate table reading, label discovery, sample-path derivation, metadata joining, one ordered owned-column tuple, and synchronous manifest writing in `training_data/manifest.py`. Do not create a top-level table utility.

- [ ] **Step 6: Implement dense orchestration**

Move the flow and submitted task into `training_data/dense.py`; pass the complete `ModelSpec` to `prepare_dense_sample`, call `model.load_rasters(anchor)`, validate metadata before submission, preserve the existing bounded loop, and call manifest writing synchronously.

- [ ] **Step 7: Update CLI and generated scripts**

Add `metadata: Path | None = None`, forward it as a string or `None`, import from `workflow.training_data`, and update workspace-generation assertions.

- [ ] **Step 8: Run focused tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/training_data tests/cli/commands/test_workflow.py tests/workflow/test_ingest_script.py`

Expected: PASS.

- [ ] **Step 9: Commit the slice**

```bash
git add src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow src/geosave_engine/templates tests/workflow tests/cli/commands/test_workflow.py
git commit -m "refactor: group dense training data preparation"
```

### Task 6: Convert release into an owned package

**Files:**
- Create: `src/geosave_engine/release/__init__.py`
- Create: `src/geosave_engine/release/artifact.py`
- Move: `src/geosave_engine/ml/huggingface.py` to `src/geosave_engine/release/huggingface.py`
- Delete: `src/geosave_engine/release.py`
- Move: `tests/test_release.py` to `tests/release/test_artifact.py`
- Move: `tests/ml/test_huggingface.py` to `tests/release/test_huggingface.py`

**Interfaces:**
- Consumes: `ModelChain`, `ModelSpec`, Hugging Face Hub, and the optional Transformers adapter.
- Produces: unchanged root imports `from geosave_engine.release import load_model, load_spec, publish_model, save_model`.

- [ ] **Step 1: Move tests and add import-laziness coverage**

In a fresh Python process, block `transformers` imports, import `geosave_engine.release`, and load a local `model_spec.yaml`; assert success and that `transformers` is absent from `sys.modules`. Retain local round trip, exact checkpoint matching, version rejection, complete-file, and Hub-call tests.

- [ ] **Step 2: Run release tests and verify import failures**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/release`

Expected: target package is absent.

- [ ] **Step 3: Build the package without changing the public interface**

Move release bundle operations into `artifact.py`, move the Transformers classes and registration into `huggingface.py`, and re-export only the four existing functions from `release/__init__.py`. Keep `GeoSaveModel` imports local inside artifact model save/load operations.

- [ ] **Step 4: Update internal imports**

Replace `geosave_engine.ml.huggingface` with `geosave_engine.release.huggingface`; do not leave an ML forwarding module or add an MLflow adapter.

- [ ] **Step 5: Run focused tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/release`

Expected: PASS.

- [ ] **Step 6: Commit the slice**

```bash
git add src/geosave_engine/release src/geosave_engine/ml tests/release tests/test_release.py tests/ml/test_huggingface.py
git commit -m "refactor: make release an artifact package"
```

### Task 7: Remove stale paths and verify the complete cleanup

**Files:**
- Modify: `src/geosave_engine/workflow/__init__.py`
- Modify: `docs/guides/workflows.md`
- Modify: affected templates and tests found by stale-path search
- Delete: empty legacy workflow package directories and obsolete tests

**Interfaces:**
- Consumes: all interfaces produced by Tasks 1-6.
- Produces: one coherent public tree with no old-path imports or duplicate execution paths.

- [ ] **Step 1: Search for stale interfaces**

Run:

```bash
rg -n 'geosave_engine\.ml\.cli|geosave_engine\.ml\.huggingface|workflow\.(flows|tasks|configs|specs)|workflow/(flows|tasks|configs|specs)|model_inputs|postprocessing|DenseAggregationSpec|TileSpec|predict_dense|BatchRunner' src tests docs/guides
```

Expected: only deliberate historical prose, if any; no executable imports, current templates, or active guide examples.

- [ ] **Step 2: Update public documentation and remove empty legacy paths**

Document `geosave_engine.model_spec`, `workflow.ingestion`, and `workflow.training_data`; remove stale active examples and empty packages without adding aliases.

- [ ] **Step 3: Run focused lint and type checks**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src tests`

Expected: PASS.

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run basedpyright`

Expected: PASS.

- [ ] **Step 4: Run the complete test suite**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest`

Expected: all non-excluded tests PASS.

- [ ] **Step 5: Verify packaging and whitespace**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv build`

Expected: wheel and source distribution build successfully with the new packages included.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 6: Commit final cleanup**

```bash
git add src tests docs/guides
git commit -m "docs: align public package paths"
```
