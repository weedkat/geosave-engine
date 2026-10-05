# Raster Requirements and STAC Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous workflow source configuration with model-owned raster requirements and recipe-first STAC acquisition while preserving native validation, lazy loading, and safe dense concurrency.

**Architecture:** `ModelSpec.rasters` maps stable raster names to `RasterRequirement`; each requirement validates native xarray structure, coordinates, grid, and attrs and may contain one `StacRecipe`. STAC workflows execute that recipe directly on a `GeoAnchor` target, while existing rasters bypass acquisition and use the same `select_raster` validation before preprocessing.

**Tech Stack:** Python 3.12, Pydantic 2, xarray/Dask, odc-stac, Prefect 3, Typer, pytest, Ruff, BasedPyright, Zensical

**Spec:** `docs/superpowers/specs/2026-09-27-raster-requirements-stac-recipe-design.md`

## Global Constraints

- `source` names only concrete external providers such as `StacSource`; model declarations use `rasters`.
- `RasterRequirement` remains lazy and accepts either named variables or positional channels.
- `coordinates` requires presence only; it does not reorder, convert, or compute coordinate arrays.
- `attrs` remains the existing optional root/data-variable/coordinate metadata contract.
- A STAC recipe is optional for native raster validation but required by STAC-backed workflows.
- Recipe bbox/intersects/datetime/IDs take priority; the target supplies only omitted search extent/time and always owns the output geobox.
- Remove `SourceConfig`, flow `sources` parameters, CLI `--sources`, and `load_raster` without compatibility aliases.
- Dense `max_concurrency` remains a per-run complete-sample limit with safe default `1`.
- Preserve atomic local Zarr publication, completed-sample reuse, fail-fast submission, and manifest-last publication.
- Preserve unrelated working-tree changes; omit any task commit whose staged diff would capture them.
- Public APIs use concise Google-style docstrings. Ignore notebooks.

## Review Focus

- A model with multiple rasters where one lacks `stac` must fail before any endpoint is opened or sample task is submitted; Task 2 adds this preflight test.
- Recipe `bbox` and `intersects` together must fail, IDs must not gain target bounds, and explicit datetime must beat target time; Task 2 adds native query tests.
- A dimension without a coordinate array must fail `coordinates` validation without Dask computation, while extra coordinates remain valid; Task 1 adds these tests.
- `load.bands` must not silently disagree with named requirement variables, while positional channels may use an explicit ordered band list; Task 1 adds both cases.
- Callables, infinities, and other non-YAML values inside `stac.load` must fail model-spec validation; Task 1 migrates the primitive-load tests.

---

### Task 1: Define the raster contract and model-owned STAC recipe

**Files:**
- Create: `src/geosave_engine/workflow/specs/rasters.py`
- Create: `src/geosave_engine/workflow/specs/stac.py`
- Modify: `src/geosave_engine/workflow/specs/model.py`
- Modify: `src/geosave_engine/workflow/specs/__init__.py`
- Modify: `src/geosave_engine/workflow/configs/source.py`
- Modify: `src/geosave_engine/workflow/configs/__init__.py`
- Delete: `src/geosave_engine/workflow/specs/sources.py`
- Create: `tests/workflow/specs/test_rasters.py`
- Create: `tests/workflow/specs/test_stac.py`
- Modify: `tests/workflow/specs/test_model.py`
- Modify: `tests/workflow/specs/test_examples.py`
- Delete: `tests/workflow/specs/test_sources.py`
- Modify: `tests/workflow/configs/test_source.py`
- Modify: `tests/workflow/tasks/test_process.py`
- Modify: `src/geosave_engine/workflow/tasks/process.py`
- Modify: `src/geosave_engine/workflow/flows/ingest.py`
- Modify: `src/geosave_engine/workflow/flows/prepare_dense_data.py`
- Modify: `tests/workflow/flows/test_ingest.py`
- Modify: `tests/workflow/flows/test_prepare_dense_data.py`
- Modify: `tests/workflow/conftest.py`
- Modify: `tests/workflow/specs/fixtures/values.yaml`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml`
- Modify: `tests/cli/core/test_workspace.py`

**Interfaces:**
- Consumes: Existing `SpecModel`, `AttrsRequirement` implementation, native `StacSourceConfig`, and `StacQuery`.
- Produces: `ModelSpec.rasters: dict[Name, RasterRequirement]`; `RasterRequirement.coordinates`; `RasterRequirement.stac`; `StacRecipe`; model-owned `QueryConfig` and `SortConfig` in `workflow.specs.stac`.

- [ ] **Step 1: Write failing coordinate and recipe tests**

  Move the existing raster requirement coverage into `test_rasters.py`. Add tests named `test_required_coordinates_must_exist_without_computing`, `test_extra_coordinates_remain_valid`, and `test_coordinate_attrs_still_apply`. In `test_stac.py`, cover valid recipe serialization, unique HTTP endpoints, conflicting bbox/intersects, named-variable/load-band agreement, positional-channel bands, and rejection of callable or non-finite load values.

- [ ] **Step 2: Write failing model-schema migration tests**

  Update `test_model.py` to require `rasters` in saved YAML and reject `sources` as an extra field. Update preprocessing tests so raw `xr.Dataset` values are selected and validated through `model.rasters` without requiring a STAC recipe.

- [ ] **Step 3: Run the spec tests to verify the new contract fails**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync pytest tests/workflow/specs/test_rasters.py tests/workflow/specs/test_stac.py tests/workflow/specs/test_model.py tests/workflow/tasks/test_process.py -q`

  Expected: FAIL because `ModelSpec.rasters`, `StacRecipe`, and coordinate presence validation do not exist.

- [ ] **Step 4: Implement the model-owned declarations**

  Move raster/attrs requirement declarations into `specs/rasters.py`. Add `coordinates: tuple[Text, ...] = ()`, unique-name validation, post-selection coordinate presence validation, and `stac: StacRecipe | None = None`. Move `SortConfig` and `QueryConfig` to `specs/stac.py` as `SpecModel` types and add `StacRecipe(collection, endpoints, query, load)` with primitive-load validation. Validate named-variable/load-band consistency across `RasterRequirement` and its recipe in the raster model.

- [ ] **Step 5: Migrate `ModelSpec` and live model declarations**

  Replace `ModelSpec.sources` with `ModelSpec.rasters`; update `preprocess` to validate consumed names through `model.rasters`; migrate every live model YAML, fixture, constructor, flow lookup, generated-workspace assertion, and test helper. Keep the transitional flow `sources` arguments and `SourceConfig` class only until Task 2, importing its `QueryConfig` from `workflow.specs.stac`, so the full workflow suite remains runnable between tasks.

- [ ] **Step 6: Run focused schema and raw-raster coverage**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync pytest tests/workflow tests/cli -q`

  Expected: PASS.

- [ ] **Step 7: Run scoped static checks**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync ruff check src/geosave_engine/workflow/specs src/geosave_engine/workflow/tasks/process.py tests/workflow/specs tests/workflow/tasks/test_process.py`

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync basedpyright src/geosave_engine/workflow/specs src/geosave_engine/workflow/tasks/process.py tests/workflow/specs tests/workflow/tasks/test_process.py`

  Expected: no diagnostics.

- [ ] **Step 8: Commit the schema migration when safe**

  Stage only Task 1 files and inspect `git diff --cached`. If they do not contain unrelated prior edits, commit with `refactor: model raster requirements and STAC recipes`; otherwise leave them uncommitted and record the overlap.

### Task 2: Execute recipes through loaders, flows, and commands

**Files:**
- Modify: `src/geosave_engine/geodata/stac/source.py`
- Modify: `src/geosave_engine/geodata/stac/query.py`
- Modify: `tests/geodata/stac/test_source.py`
- Modify: `src/geosave_engine/workflow/tasks/load.py`
- Modify: `src/geosave_engine/workflow/tasks/dense.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`
- Delete: `src/geosave_engine/workflow/configs/source.py`
- Modify: `src/geosave_engine/workflow/configs/__init__.py`
- Modify: `tests/workflow/tasks/test_load.py`
- Modify: `tests/workflow/tasks/test_dense.py`
- Delete: `tests/workflow/configs/test_source.py`
- Modify: `src/geosave_engine/workflow/flows/ingest.py`
- Modify: `src/geosave_engine/workflow/flows/prepare_dense_data.py`
- Modify: `tests/workflow/flows/test_ingest.py`
- Modify: `tests/workflow/flows/test_prepare_dense_data.py`
- Modify: `src/geosave_engine/cli/commands/workflow.py`
- Modify: `tests/cli/commands/test_workflow.py`
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Modify: `tests/workflow/test_ingest_script.py`

**Interfaces:**
- Consumes: Task 1 `ModelSpec.rasters`, `RasterRequirement.stac`, `StacRecipe`, `QueryConfig`, and existing `StacSource`/`StacClient` capabilities.
- Produces: `load_stac_raster(target: GeoAnchor, requirement: RasterRequirement) -> xr.Dataset`; source-free `ingest` and `prepare_dense_data` signatures from the spec; CLI commands without `--sources`.

- [ ] **Step 1: Write failing recipe-first native query tests**

  In `tests/geodata/stac/test_source.py`, assert `_search_query` behavior through the public search client: explicit bbox, intersects, and datetime survive `load(target)`; omitted extent/time come from the target; IDs receive no target-derived selectors; conflicting bbox/intersects fail before search.

- [ ] **Step 2: Write failing loader and dense-task tests**

  Rename loader tests around `load_stac_raster(target, requirement)`. Assert it requires `requirement.stac`, applies recipe query/load settings, preserves lazy pixels, selects required variables, retries fallback endpoints, and rejects malformed results. Update `_prepare_dense_sample(label, requirements, output)` tests so no independent config mapping is accepted.

- [ ] **Step 3: Write failing flow and CLI migration tests**

  Remove `sources` from flow calls and expected CLI arguments. Assert CLI help has no `--sources`; both flows read recipes from `ModelSpec.rasters`; and a model with two rasters where either recipe is missing fails before `Client.open`, `_prepare_dense_sample.submit`, or anchor raster opening. Retain dense concurrency, failure, resume, and manifest assertions unchanged.

- [ ] **Step 4: Run the acquisition tests to verify the old path fails**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync pytest tests/geodata/stac/test_source.py tests/workflow/tasks/test_load.py tests/workflow/tasks/test_dense.py tests/workflow/flows/test_ingest.py tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py -q`

  Expected: FAIL because target fallback still overrides recipe spatial queries and the runtime `SourceConfig` path still exists.

- [ ] **Step 5: Implement recipe-first native search behavior**

  Make native `StacQuery` reject simultaneous bbox/intersects. Change `StacSource._search_query(anchor)` to return an IDs query unchanged; otherwise preserve explicit bbox/intersects/datetime and fill only absent extent/time from the target. Keep the target geobox passed to `odc.stac.load` regardless of search selectors.

- [ ] **Step 6: Implement `load_stac_raster` and remove runtime source configuration**

  Replace `load_raster(anchor, source, requirement)` with `load_stac_raster(target, requirement)`. Configure the fresh `StacSource` solely from `requirement.stac`, retain cached clients and fallback endpoints, then return `requirement.select_raster(stac.load(target))`. Update `_prepare_dense_sample` accordingly; delete `SourceConfig` and its exports/tests.

- [ ] **Step 7: Migrate flows, commands, and the generated script**

  In both flows, load `model.rasters`, collect every name lacking `stac`, and reject the complete list before opening anchors or submitting tasks. Remove flow `sources` parameters, CLI JSON source parsing and `--sources`, the boilerplate `SOURCES` constant, and all forwarding assertions. Keep the approved ordinary options and dense concurrency behavior.

- [ ] **Step 8: Run focused default and slow behavior**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync pytest tests/geodata/stac/test_source.py tests/workflow/tasks/test_load.py tests/workflow/tasks/test_dense.py tests/workflow/flows/test_ingest.py tests/workflow/flows/test_prepare_dense_data.py tests/cli/commands/test_workflow.py tests/workflow/test_ingest_script.py -q`

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py -m slow -q`

  Expected: PASS.

- [ ] **Step 9: Run scoped static checks**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync ruff check src/geosave_engine/geodata/stac/source.py src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow.py tests/geodata/stac/test_source.py tests/workflow tests/cli/commands/test_workflow.py`

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync basedpyright src/geosave_engine/geodata/stac/source.py src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow.py tests/geodata/stac/test_source.py tests/workflow tests/cli/commands/test_workflow.py`

  Expected: no diagnostics.

- [ ] **Step 10: Commit recipe-driven execution when safe**

  Stage only Task 2 files and inspect `git diff --cached`. Commit with `refactor: execute model STAC recipes` only when the staged diff contains no unrelated prior edits.

### Task 3: Update the guide and verify the breaking migration

**Files:**
- Modify: `docs/guides/workflows.md`
- Modify: live non-notebook references found by the migration search

**Interfaces:**
- Consumes: Task 2 final Python, YAML, flow, and CLI surfaces.
- Produces: User documentation for model-owned recipes and commands without runtime source configuration.

- [ ] **Step 1: Rewrite workflow examples around model-owned recipes**

  Document the `rasters` schema, coordinate and attrs requirements, nested STAC recipe, recipe-first target fallback, existing-raster validation path, and source-free CLI commands. Keep the explanation of `max_concurrency=1` as complete simultaneous sample ingestion rather than requests per second.

- [ ] **Step 2: Search for obsolete live vocabulary**

  Run: `rg -n "ModelSpec\.sources|\.sources\[|sources:|SourceConfig|load_raster|--sources|sources=" src tests docs --glob '!*.ipynb' --glob '!docs/superpowers/**'`

  Expected: no obsolete live API or YAML references. Update genuine live references; historical design documents remain unchanged.

- [ ] **Step 3: Run the complete affected behavioral suites**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync pytest tests/geodata/stac tests/workflow tests/cli -q`

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync pytest tests/workflow -m slow -q`

  Expected: PASS.

- [ ] **Step 4: Run static and documentation verification**

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync ruff check src/geosave_engine/geodata/stac src/geosave_engine/workflow src/geosave_engine/cli src/geosave_engine/templates tests/geodata/stac tests/workflow tests/cli`

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync basedpyright src/geosave_engine/geodata/stac src/geosave_engine/workflow src/geosave_engine/cli tests/geodata/stac tests/workflow tests/cli/commands`

  Run: `UV_CACHE_DIR=/tmp/geosave-raster-recipe-cache uv run --offline --no-sync zensical build --clean`

  Run: `git diff --check`

  Expected: no diagnostics, documentation build success, and no whitespace errors. If broad BasedPyright still reports only the pre-existing optional-value diagnostics in `tests/cli/core/test_workspace.py`, record them and rerun the plan-owned scope separately.

- [ ] **Step 5: Commit documentation and remaining safe migration edits**

  Stage only Task 3 files and inspect the staged diff. Commit with `docs: document model STAC recipes` only when it contains no unrelated prior edits; otherwise leave overlapping edits uncommitted and report them.
