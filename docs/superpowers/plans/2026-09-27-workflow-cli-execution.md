# Workflow CLI and Execution Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `ingest` and corruption-safe dense data preparation as local Typer commands while making preprocessing and postprocessing single Prefect tasks.

**Architecture:** Only jobs with serializable inputs and durable outputs remain in `workflow.flows`. Model processing moves to sequential tasks, dense sample work moves to `workflow.tasks`, and `prepare_dense_data` owns a bounded in-flight queue whose safe default is one. The CLI maps typed options and Pydantic-validated JSON directly onto those flow signatures.

**Tech Stack:** Python 3.12, Typer, Pydantic 2, Prefect 3, xarray/Dask, pytest, Ruff, BasedPyright, Zensical

**Spec:** `docs/superpowers/specs/2026-09-27-workflow-cli-execution-design.md`

## Global Constraints

- `model_spec.yaml` remains the only public workflow YAML document.
- Do not add compatibility aliases, a flow registry, generic `--param` parsing, or a second job configuration file.
- `max_concurrency` defaults to `1` and limits complete sample ingestions from STAC search through Zarr persistence.
- Existing local Zarr atomic publication and completed-sample reuse behavior must remain intact.
- Public APIs use concise Google-style docstrings, and tests mirror source ownership.
- Preserve unrelated working-tree changes; before any commit, inspect the staged diff and omit the commit if it would capture pre-existing unrelated edits.
- Ignore notebooks.

## Review Focus

- Malformed or correctly formed but wrongly shaped `--anchor` and `--sources` JSON must fail before invoking a flow; Task 4 adds these CLI tests.
- Omitting `sources` for a model with multiple source requirements must create one default `SourceConfig` per exact source name; Task 3 adds this flow test.
- `max_concurrency=0` must fail before label tasks are submitted; Task 3 adds this validation test.
- A failed sample with other work in flight must stop further submission and must not replace the existing manifest; Task 3 adds this scheduling test.
- A runtime flow exception must make the CLI exit nonzero without printing a success path; Task 4 adds this command test.

---

### Task 1: Collapse model processing flows into tasks

**Files:**
- Create: `src/geosave_engine/workflow/tasks/process.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`
- Modify: `src/geosave_engine/workflow/flows/__init__.py`
- Delete: `src/geosave_engine/workflow/tasks/call.py`
- Delete: `src/geosave_engine/workflow/flows/preprocess.py`
- Delete: `src/geosave_engine/workflow/flows/postprocess.py`
- Delete: `src/geosave_engine/workflow/flows/stage.py`
- Create: `tests/workflow/tasks/test_process.py`
- Delete: `tests/workflow/tasks/test_call.py`
- Delete: `tests/workflow/flows/test_preprocess.py`
- Delete: `tests/workflow/flows/test_postprocess.py`
- Modify: `tests/workflow/specs/test_examples.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py`

**Interfaces:**
- Consumes: `ModelSpec.preprocessing`, `ModelSpec.postprocessing`, `StageSpec.validate_inputs(names)`, `CallSpec.select_inputs(state)`, and `CallSpec.invoke(inputs)`.
- Produces: `preprocess(inputs: Mapping[str, Any], spec: ModelSpec) -> dict[str, Any]` and `postprocess(inputs: Mapping[str, Any], spec: ModelSpec) -> dict[str, Any]`, each decorated once with Prefect `@task(cache_policy=NO_CACHE, persist_result=False)`.

- [x] **Step 1: Write the failing task-boundary tests**

  Create `tests/workflow/tasks/test_process.py` by migrating the durable preprocessing and postprocessing cases. Add assertions that both exports are Prefect `Task` instances, processing calls run in declaration order, independent declarations no longer overlap, validation happens before the first call, source requirements are selected only for consumed preprocessing inputs, rebinding and `None` results work, and only declared outputs are returned. Assert `workflow.flows` no longer exposes `preprocess`, `postprocess`, or `run_stage` and `workflow.tasks` no longer exposes `invoke_call`.

- [x] **Step 2: Run the processing tests to verify the new boundary fails**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_process.py -q`

  Expected: FAIL because `workflow.tasks.process` and its task exports do not exist.

- [x] **Step 3: Implement the sequential processing tasks**

  Add a private `_run_stage(stage: StageSpec, inputs: Mapping[str, Any]) -> dict[str, Any]` in `tasks/process.py`. It validates references before invocation, copies state, executes each declaration through `CallSpec.invoke` in order, binds results for later references, and returns declared outputs. Implement `preprocess` with consumed-source requirement selection and `postprocess` as the direct stage wrapper. Export both tasks and remove the obsolete flow/task modules and exports.

- [x] **Step 4: Migrate live examples to the task imports**

  Change template code, template documentation, and `test_examples.py` to import `preprocess` and `postprocess` from `geosave_engine.workflow.tasks`. Keep their model-spec examples and result assertions unchanged.

- [x] **Step 5: Run focused processing and example tests**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_process.py tests/workflow/specs/test_examples.py tests/cli/core/test_workspace.py -q`

  Expected: PASS.

- [x] **Step 6: Run scoped static checks**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync ruff check src/geosave_engine/workflow/tasks src/geosave_engine/workflow/flows src/geosave_engine/templates/tasks/semantic_segmentation tests/workflow/tasks/test_process.py tests/workflow/specs/test_examples.py`

  Expected: no diagnostics.

- [x] **Step 7: Commit the processing boundary when safe**

  Stage only the files named in this task and inspect `git diff --cached`. If no pre-existing unrelated edits are included, commit with `refactor: make model processing workflow tasks`; otherwise leave the task changes uncommitted for the final handoff.

### Task 2: Separate dense sample work from orchestration

**Files:**
- Create: `src/geosave_engine/workflow/tasks/dense.py`
- Modify: `src/geosave_engine/workflow/dense.py`
- Modify: `src/geosave_engine/workflow/configs/source.py`
- Modify: `src/geosave_engine/workflow/tasks/load.py`
- Create: `tests/workflow/tasks/test_dense.py`
- Modify: `tests/workflow/configs/test_source.py`
- Modify: `tests/workflow/tasks/test_load.py`
- Modify: `tests/workflow/test_dense.py`

**Interfaces:**
- Consumes: `load_raster(anchor, source, requirement) -> xr.Dataset`, `write_stack(rasters, output) -> str`, and the existing model source requirements.
- Produces: private `_validate_dense_sample(path, requirements) -> None` and Prefect task `_prepare_dense_sample(label, sources, requirements, output) -> str` in `workflow.tasks.dense`; `SourceConfig` contains only `query` and `load`.

- [x] **Step 1: Write failing dense-task and source-config tests**

  Add `tests/workflow/tasks/test_dense.py` for label-derived anchors, complete sample persistence, missing label time, valid sample reuse without STAC access, and rejection of incomplete existing samples. Update the source-config test to assert `SourceConfig.model_validate({"concurrency": "cdse"})` raises for an extra field. Update load tests to assert `source_concurrency` is absent instead of provisioning Prefect global limits.

- [x] **Step 2: Run the focused tests to verify they fail**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_dense.py tests/workflow/configs/test_source.py tests/workflow/tasks/test_load.py -q`

  Expected: FAIL because dense sample ownership has not moved and `SourceConfig.concurrency` still exists.

- [x] **Step 3: Implement dense sample ownership and remove named limits**

  Move sample validation and the per-label task from `workflow/dense.py` into `workflow/tasks/dense.py`, preserving atomic `write_stack` behavior and completed-sample reuse. Remove `SourceConfig.concurrency`, its docstring entry, the `source_concurrency` context manager, and all Prefect concurrency imports. Make the transitional root dense flow call `_prepare_dense_sample` directly without a concurrency context so behavior remains testable until Task 3 moves the flow.

- [x] **Step 4: Run dense-task, loading, and persistence tests**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow/tasks/test_dense.py tests/workflow/tasks/test_load.py tests/workflow/tasks/test_save.py tests/workflow/configs/test_source.py -q`

  Expected: PASS.

- [x] **Step 5: Commit the dense task boundary when safe**

  Stage only Task 2 files, inspect the staged diff, and commit with `refactor: separate dense sample task` only when it contains no unrelated prior edits.

### Task 3: Expose bounded runnable flows

**Files:**
- Create: `src/geosave_engine/workflow/flows/prepare_dense_data.py`
- Modify: `src/geosave_engine/workflow/flows/ingest.py`
- Modify: `src/geosave_engine/workflow/flows/__init__.py`
- Modify: `src/geosave_engine/workflow/__init__.py`
- Delete: `src/geosave_engine/workflow/dense.py`
- Create: `tests/workflow/flows/test_prepare_dense_data.py`
- Modify: `tests/workflow/flows/test_ingest.py`
- Delete: `tests/workflow/test_dense.py`
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`
- Modify: `tests/workflow/test_ingest_script.py`

**Interfaces:**
- Consumes: Task 2 `_prepare_dense_sample`, `write_manifest`, `SourceConfig`, `ModelSpec`, and Prefect futures.
- Produces: `ingest(anchor, *, output, spec, sources=None) -> str` and `prepare_dense_data(labels, *, output, spec, pattern="**/*.tif", sources=None, max_concurrency=1) -> str`, exported only from `workflow.flows`.

- [x] **Step 1: Write failing public-boundary and default-source tests**

  Assert `workflow.flows` exports exactly the runnable flow names plus no processing helpers, the workflow root has no `dense`, and both flows construct default runtime configs for every model source when `sources=None`. Retain tests that explicit source names must exactly match requirements.

- [x] **Step 2: Write failing bounded-scheduling tests**

  Add flow tests using controlled sample tasks and a synchronized active counter. Assert the omitted/default limit never observes more than one active sample, `max_concurrency=2` reaches two but never three, and `max_concurrency=0` fails before `_prepare_dense_sample.submit` is called. Add a failure test with more labels than available slots: after one submitted future fails, no later labels are submitted, any already completed sample remains, and an existing manifest is unchanged.

- [x] **Step 3: Run the new flow tests to verify they fail**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py tests/workflow/flows/test_ingest.py -q`

  Expected: FAIL because `prepare_dense_data` and optional source defaults do not exist.

- [x] **Step 4: Implement default source resolution and bounded submission**

  Add a small flow-local source normalization helper that validates supplied configs or creates `{name: SourceConfig()}` from `ModelSpec.sources`. In `prepare_dense_data`, validate `PositiveInt`, discover labels in stable order, submit at most `max_concurrency` futures, and use Prefect `as_completed` over the current pending set to submit one replacement after each success. On `future.result()` failure, propagate immediately without adding another task. Publish the manifest only after all sample futures succeed.

- [x] **Step 5: Complete the flow migration**

  Export `prepare_dense_data`, remove the root dense namespace and file, update `ingest` to its approved signature and default-source behavior, and migrate the generated ingest script plus its test to `workflow.flows.prepare_dense_data`. Do not add aliases.

- [x] **Step 6: Run flow, template-script, and slow concurrency coverage**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py tests/workflow/flows/test_ingest.py tests/workflow/test_ingest_script.py -q`

  Expected: PASS for default tests.

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow/flows/test_prepare_dense_data.py -m slow -q`

  Expected: PASS, with local Prefect/STAC fixtures available.

- [x] **Step 7: Commit the runnable flow boundary when safe**

  Stage only Task 3 files, inspect the staged diff, and commit with `refactor: expose dense preparation flow` only when it contains no unrelated prior edits.

### Task 4: Add the workflow Typer commands

**Files:**
- Create: `src/geosave_engine/cli/commands/workflow.py`
- Modify: `src/geosave_engine/cli/main.py`
- Create: `tests/cli/commands/test_workflow.py`

**Interfaces:**
- Consumes: Task 3 `ingest` and `prepare_dense_data`, `AnchorConfig`, `SourceConfig`, Pydantic `TypeAdapter`, and Typer.
- Produces: the `workflow` Typer group with `ingest` and `prepare-dense-data` commands; successful commands print exactly the returned path.

- [x] **Step 1: Write failing CLI help and argument-forwarding tests**

  Use `typer.testing.CliRunner` to assert the root help lists `workflow`, group help lists both commands, ingest help lists `--anchor`, `--output`, `--spec`, and `--sources`, and dense help lists `--labels`, `--output`, `--spec`, `--pattern`, `--sources`, and `--max-concurrency` with default `1`. Monkeypatch both flow objects and assert parsed arguments exactly match their approved signatures and successful output contains only the returned path.

- [x] **Step 2: Write failing structured-input and runtime-error tests**

  Parameterize malformed JSON, an anchor with an unknown `kind`, a non-mapping `--sources` value, and a source containing an unknown field. Assert exit code `2` and that neither flow is invoked. Make a patched flow raise `RuntimeError("ingestion failed")`; assert a nonzero exit and absence of a success path.

- [x] **Step 3: Run the CLI tests to verify they fail**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/cli/commands/test_workflow.py -q`

  Expected: FAIL because the workflow command group does not exist.

- [x] **Step 4: Implement the command group and shared JSON validation**

  Define a Typer sub-application in `cli/commands/workflow.py`. Add one private `_parse_json(value: str, adapter: TypeAdapter[Any], option: str) -> Any` that converts Pydantic validation failures to `typer.BadParameter` for the originating option. Validate anchors through `TypeAdapter(AnchorConfig)` and sources through `TypeAdapter(dict[str, SourceConfig])`, then dump them to primitive dictionaries before calling the flows. Mount the group with `app.add_typer(..., name="workflow")` after the global dotenv callback remains available.

- [x] **Step 5: Run CLI tests and smoke-test generated help**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/cli/commands/test_workflow.py tests/cli/core/test_workspace.py -q`

  Expected: PASS.

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync geosave workflow prepare-dense-data --help`

  Expected: exit `0`, with `--max-concurrency` documented as defaulting to `1`.

- [x] **Step 6: Commit the CLI when safe**

  Stage only Task 4 files, inspect the staged diff, and commit with `feat: add workflow commands` only when it contains no unrelated prior edits.

### Task 5: Publish the workflow guide and verify the migration

**Files:**
- Create: `docs/guides/workflows.md`
- Modify: `zensical.toml`
- Modify: any live non-notebook references found by the migration search

**Interfaces:**
- Consumes: the final Python and CLI surfaces from Tasks 1-4.
- Produces: a navigable workflow guide whose commands and defaults match Typer help.

- [x] **Step 1: Write the user guide**

  Lead with a two-row purpose/input/output comparison for `ingest` and `prepare-dense-data`. Add minimal commands, a complete option table for each command, raster/GeoJSON/coordinate anchor JSON, omitted and explicit source settings, serial safety and explicit parallel opt-in, atomic failure/resume behavior, and equivalent Python calls. Describe `max_concurrency` as complete simultaneous sample ingestions, not HTTP requests per second.

- [x] **Step 2: Add the guide to explicit Zensical navigation**

  Change `nav = []` to `nav = [{ "Workflow guide" = "guides/workflows.md" }]`. Do not add unrelated documentation pages.

- [x] **Step 3: Search for obsolete live imports and names**

  Run: `rg -n "workflow\.dense|dense\.prepare|dense\.validate_sample|flows import .*preprocess|flows import .*postprocess|run_stage|invoke_call|SourceConfig\([^\n]*concurrency|source_concurrency|prepare_training" src tests docs --glob '!*.ipynb' --glob '!superpowers/**'`

  Expected: no obsolete live API references. Update only genuine live references; historical design documents remain unchanged.

- [x] **Step 4: Run focused and broad behavioral verification**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow tests/cli -q`

  Expected: PASS under the default marker selection.

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync pytest tests/workflow -m slow -q`

  Expected: PASS with local Prefect/STAC fixtures available.

- [x] **Step 5: Run static and documentation checks**

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync ruff check src/geosave_engine/workflow src/geosave_engine/cli src/geosave_engine/templates tests/workflow tests/cli`

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync basedpyright src/geosave_engine/workflow src/geosave_engine/cli tests/workflow tests/cli`

  Run: `UV_CACHE_DIR=/tmp/geosave-workflow-cli-cache uv run --offline --no-sync zensical build --clean`

  Run: `git diff --check`

  Expected: no diagnostics, documentation build success, and no whitespace errors.

- [x] **Step 6: Commit documentation and any remaining safe migration edits**

  Inspect the complete staged diff. Commit only files attributable to this plan with `docs: add workflow execution guide`; leave all unrelated working-tree changes untouched and report any plan changes that could not be committed independently.
