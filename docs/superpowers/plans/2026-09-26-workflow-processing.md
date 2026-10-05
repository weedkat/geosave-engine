# Workflow Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify preprocessing and postprocessing on `StageSpec`, execute both through one Prefect stage runner, and expose segmentation output transforms as model-spec-callable library functions.

**Architecture:** `model_spec.yaml` keeps three lifecycle stages with the same ordered call grammar. Imported domain functions are preferred, while native xarray/accessor methods remain valid referenced bound callables; public `run_stage` composes either stage inside a Prefect flow and preserves dependency-aware execution.

**Tech Stack:** Python 3.13, Pydantic v2, PyYAML, Prefect, xarray/Dask, PyTorch/Lightning, pytest, Ruff

**Spec:** [Workflow processing design](../specs/2026-09-26-workflow-processing-design.md)

## Global Constraints

- Keep `ModelSpec.filename` as a class attribute whose default is exactly `"model_spec.yaml"`; `resolve_path` is a staticmethod and uses that attribute.
- Keep `CallSpec.call` as `str | Ref`; Python binds `self` for referenced methods and all remaining arguments come from `kwargs`.
- Do not add registries, expression syntax, generic method wrappers, compatibility aliases, or a second execution path.
- Keep raster preprocessing lazy and preserve the current independent-call concurrency.
- Model loading, tile accumulation, batch lifetime, persistence, and the complete prediction flow remain out of scope.
- The ML relocation under `src/geosave_engine/ml/lightning/` is pre-existing untracked user work. Do not stage or commit implementation changes from this plan.

## Review Focus

- A referenced bound method must receive Python-bound `self` and only its declared kwargs; Task 1 retains the selection/method invocation regression test.
- A postprocessing declaration referencing `task.class_thresholds` and `task.ignore_index` must select the `task` root once and preserve those object identities; Task 2 tests task-state resolution.
- Omitting optional `mask` must invoke the transform normally, while declaring `!ref mask` without supplying it must fail before task submission; Task 2 tests both forms.
- Thresholded labels must be `torch.uint8`, probabilities `torch.float32`, and logits must remain unchanged; Task 3 owns these assertions.
- Extracting the runner must not compute a lazy raster or validate an unused source; Task 4 reruns the shipped lazy preprocessing example.

---

### Task 1: Unify the model processing declarations

**Files:**
- Modify: `src/geosave_engine/workflow/specs/model.py:12-94`
- Modify: `src/geosave_engine/workflow/specs/__init__.py:1-29`
- Delete: `src/geosave_engine/workflow/specs/postprocessing.py`
- Modify: `tests/workflow/specs/test_model.py`

**Interfaces:**
- Consumes: Existing `StageSpec`, `CallSpec`, `Ref`, and strict YAML loader/dumper.
- Produces: `ModelSpec.preprocessing`, `.inference`, and `.postprocessing` as `StageSpec`; `ModelSpec.resolve_path(path: str | Path) -> Path` as a staticmethod.

- [ ] **Step 1: Write failing model-spec tests**

Replace the reserved-postprocessing test with assertions that all three fields are `StageSpec`, that postprocessing accepts an ordered `CallSpec`, and that YAML round trips it. Add:

```python
def test_resolve_path_is_static_and_uses_model_filename(tmp_path):
    descriptor = inspect.getattr_static(ModelSpec, "resolve_path")
    assert isinstance(descriptor, staticmethod)
    assert ModelSpec.filename == "model_spec.yaml"
    assert ModelSpec.resolve_path(tmp_path) == tmp_path / ModelSpec.filename
```

Keep the existing referenced-bound-method coverage; it proves the corrected design still supports xarray-style selection.

- [ ] **Step 2: Run the focused test and verify the new expectations fail**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/specs/test_model.py -q`

Expected: FAIL because postprocessing is still `PostprocessingSpec` and `resolve_path` is still a classmethod.

- [ ] **Step 3: Implement the minimal model-spec cleanup**

In `ModelSpec`, set:

```python
postprocessing: StageSpec = Field(default_factory=StageSpec)

@staticmethod
def resolve_path(path: str | Path) -> Path:
```

Use `ModelSpec.filename` for directory/default resolution. Remove the `PostprocessingSpec` import, file, and public export. Do not change `CallSpec` target semantics.

- [ ] **Step 4: Run the model-spec tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/specs/test_model.py tests/workflow/specs/test_call.py -q`

Expected: PASS.

- [ ] **Step 5: Checkpoint without staging user work**

Run: `git diff --check -- src/geosave_engine/workflow/specs tests/workflow/specs/test_model.py`

Expected: no output.

### Task 2: Share stage execution and add postprocessing orchestration

**Files:**
- Create: `src/geosave_engine/workflow/flows/stage.py`
- Create: `src/geosave_engine/workflow/flows/postprocess.py`
- Modify: `src/geosave_engine/workflow/flows/preprocess.py:1-45`
- Modify: `src/geosave_engine/workflow/flows/__init__.py:1-6`
- Create: `tests/workflow/flows/test_postprocess.py`
- Modify: `tests/workflow/flows/test_preprocess.py`

**Interfaces:**
- Consumes: `StageSpec.validate_inputs`, `CallSpec.select_inputs`, and Prefect `invoke_call`.
- Produces: public `run_stage(stage: StageSpec, inputs: Mapping[str, Any], *, name: str) -> dict[str, Any]` and `postprocess(inputs: Mapping[str, Any], spec: ModelSpec) -> dict[str, Any]`.

- [ ] **Step 1: Write failing postprocessing-flow tests**

Cover:

```python
def test_empty_postprocessing_returns_no_supplied_values(prefect_server): ...
def test_postprocessing_resolves_preceding_results_and_task_state(prefect_server): ...
def test_postprocessing_omits_optional_mask(prefect_server): ...
def test_postprocessing_rejects_a_declared_missing_mask_before_submission(monkeypatch): ...
def test_postprocessing_names_prefect_tasks(prefect_server): ...
```

Use a small task-state object whose `class_thresholds` and `ignore_index` identities can be asserted by an imported test helper. Assert that only declared outputs are returned and a later declaration can reference an earlier result.

- [ ] **Step 2: Run the postprocessing-flow tests and verify they fail**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/flows/test_postprocess.py -q`

Expected: FAIL because the flow and shared runner do not exist.

- [ ] **Step 3: Extract the private runner and implement the flow**

Implement `run_stage` to validate all stage inputs before configuring tasks, copy the supplied mapping, submit `invoke_call` under `{name}-{output}` task names, bind futures for downstream declarations, and return only declared results. Export it from `workflow.flows` for composition inside custom Prefect flows. Refactor `preprocess` to select consumed raster sources and then call `run_stage(name="preprocess")`. Implement `postprocess` as a validated-copy wrapper around `run_stage(name="postprocess")`.

- [ ] **Step 4: Run preprocessing and postprocessing flow tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/flows/test_preprocess.py tests/workflow/flows/test_postprocess.py -q`

Expected: PASS, including validation-before-submission and independent-call overlap.

- [ ] **Step 5: Checkpoint without staging user work**

Run: `git diff --check -- src/geosave_engine/workflow/flows tests/workflow/flows`

Expected: no output.

### Task 3: Move segmentation interpretation to ML transforms

**Files:**
- Create: `src/geosave_engine/ml/transforms/semantic_segmentation.py`
- Modify: `src/geosave_engine/ml/transforms/__init__.py`
- Modify: `src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py:1-245`
- Modify: `src/geosave_engine/ml/callbacks/threshold_calibrator.py:1-16`
- Modify: `tests/ml/callbacks/test_task_callbacks.py:1-14`
- Delete: `tests/ml/tasks/test_segmentation_postprocess.py`
- Create: `tests/ml/transforms/test_semantic_segmentation.py`

**Interfaces:**
- Consumes: Raw `[B, C, H, W]` logits and task-owned threshold/ignore state.
- Produces: `softmax_argmax(logits) -> tuple[Tensor, Tensor]` and `apply_thresholds(logits, thresholds, ignore_index, mask=None) -> tuple[Tensor, Tensor]` from `geosave_engine.ml.transforms.semantic_segmentation`.

- [ ] **Step 1: Move and strengthen the transform test before production edits**

Import from the new module and assert the existing threshold/nodata values plus:

```python
assert result.dtype == torch.uint8
assert scores.dtype == torch.float32
torch.testing.assert_close(logits, original)
```

- [ ] **Step 2: Run the transform test and verify it fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/transforms/test_semantic_segmentation.py -q`

Expected: FAIL because the transform module does not exist.

- [ ] **Step 3: Move the functions and their complete output contract**

Move both functions to `ml.transforms.semantic_segmentation`, cast labels and probabilities before returning from `apply_thresholds`, and export them from `ml.transforms`.

- [ ] **Step 4: Remove the duplicate task method and update consumers**

Remove `SemanticSegmentationTask.postprocess` and update its prediction docstring to refer to model-spec postprocessing instead of a task method. Import `softmax_argmax` from `geosave_engine.ml.transforms.semantic_segmentation` in `ThresholdCalibrator`. Update the callback test's task import to `geosave_engine.ml.lightning.tasks` to match the in-progress source relocation.

- [ ] **Step 5: Run transform and callback-focused tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/transforms/test_semantic_segmentation.py tests/ml/callbacks/test_task_callbacks.py -q`

Expected: PASS.

- [ ] **Step 6: Checkpoint without staging user work**

Run: `git diff --check -- src/geosave_engine/ml tests/ml/transforms`

Expected: no output.

### Task 4: Publish and exercise the complete model-owned recipe

**Files:**
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model.yaml:20`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md:1-58`
- Modify: `tests/cli/core/test_workspace.py:1-16`
- Modify: `tests/workflow/specs/test_examples.py`

**Interfaces:**
- Consumes: Importable geodata transforms, native `.gs.to_tensor`, the new `postprocess` flow, stitched logits, and task state.
- Produces: A shipped model spec whose preprocessing, inference conversion, and postprocessing all demonstrate the approved call/reference rules.

- [ ] **Step 1: Write failing shipped-example assertions**

Assert preprocessing calls `geosave_engine.geodata.transform.nodata.to_nan` and `geosave_engine.geodata.transform.packing.unpack` with predecessor values in kwargs; inference retains `Ref("image.gs.to_tensor")`; postprocessing calls `geosave_engine.ml.transforms.semantic_segmentation.apply_thresholds` with `logits`, `task.class_thresholds`, and `task.ignore_index` references. Add an execution test using real logits and a small task-state object.

- [ ] **Step 2: Run the shipped-example tests and verify they fail**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/specs/test_examples.py -q`

Expected: FAIL because preprocessing still uses bound accessors and postprocessing is empty.

- [ ] **Step 3: Update the model YAML and its purpose/input/output documentation**

Use imported geodata transform calls with `data: !ref ...`, retain the native tensor bound method, and add the segmentation transform declaration. Update the README to state that postprocessing turns stitched logits into uint8 labels and float32 probabilities; keep model loading, inference, tiling, and merging explicitly deferred. Update the template model class path and workspace test import to `geosave_engine.ml.lightning.tasks.SemanticSegmentationTask`, matching the source location already present in the worktree.

- [ ] **Step 4: Run shipped-example and workspace tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/workflow/specs/test_examples.py tests/cli/core/test_workspace.py -q`

Expected: PASS; preprocessing remains lazy and the workspace copy loads the same declarations.

- [ ] **Step 5: Checkpoint without staging user work**

Run: `git diff --check -- src/geosave_engine/templates tests/workflow/specs/test_examples.py`

Expected: no output.

### Task 5: Focused verification

**Files:**
- Verify only; do not stage or commit the dirty worktree.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: Evidence that the unified schema, both flows, ML transforms, and shipped example agree.

- [ ] **Step 1: Run the focused behavioral suite**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest \
  tests/workflow/specs/test_call.py \
  tests/workflow/specs/test_model.py \
  tests/workflow/specs/test_examples.py \
  tests/workflow/flows/test_preprocess.py \
  tests/workflow/flows/test_postprocess.py \
  tests/ml/transforms/test_semantic_segmentation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run focused lint**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check \
  src/geosave_engine/workflow \
  src/geosave_engine/ml/transforms \
  src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py \
  src/geosave_engine/ml/callbacks/threshold_calibrator.py \
  tests/workflow \
  tests/ml/transforms
```

Expected: PASS.

- [ ] **Step 3: Check the complete patch**

Run: `git diff --check`

Expected: no output. Review `git status --short` and confirm no pre-existing user change was staged, reverted, or overwritten.
