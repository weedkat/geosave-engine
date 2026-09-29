# Native Model Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish configured native PyTorch inference graphs and their independent geospatial contracts as safe, code-free Hugging Face releases.

**Architecture:** `ModelChain` remains the native `torch.nn.Module` publication unit and carries its resolved registry recipe. An installed Transformers adapter generates configuration and safetensors, while the top-level `geosave_engine.release` module independently loads models and `ModelSpec` documents. Lightning remains training-only; built-in tasks expose their deployable chain as `task.model`.

**Tech Stack:** Python 3.12, PyTorch, Lightning, Transformers, Hugging Face Hub, safetensors, Pydantic, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-09-29-native-model-release-design.md`

## Global Constraints

- A release contains exactly `config.json`, `model.safetensors`, `model_spec.yaml`, and `README.md`; it contains no executable Python.
- Publication accepts only `ModelChain` instances built by `build_model()` from registered `name` selectors; workspace `class_path` selectors remain training-only.
- `config.json` is generated with `format_version: 1` and the exact installed GeoSave version; loading rejects either mismatch.
- `load_model()` and `load_spec()` are independent and return `ModelChain` and `ModelSpec`, respectively.
- `load_spec()` must not import Transformers or download model weights.
- `publish_model()` returns the immutable Hub commit OID; production callers pass that OID as both loaders' revision.
- One-stage backbone chains are valid releases. Training-only MAE decoders and all Lightning state remain absent.
- Do not add LitServe as a dependency or move raster acquisition, tiling, aggregation, or persistence into the model.
- Preserve all unrelated working-tree changes. Stage only task-owned hunks, inspect the cached diff before every commit, and omit a task commit if existing edits cannot be isolated safely.

## Review Focus

- A missing local `Path` must raise a local file error rather than being interpreted as a Hub repository; Task 3 adds this test.
- A model input absent from `model_spec.yaml` must abort before the destination appears; Task 3 adds this atomicity test.
- A release created by a different GeoSave version must fail before inference; Task 2 adds this test.
- A mutable target branch must not be returned as the deployment identifier; Task 4 verifies the returned commit OID.
- Spec-only loading must work without importing Transformers and must request only `model_spec.yaml`; Task 3 adds both assertions.

---

### Task 1: Headless ModelChain Outputs

**Files:**
- Modify: `src/geosave_engine/ml/model_chain/chain.py`
- Test: `tests/ml/model_chain/test_chain.py`

**Interfaces:**
- Consumes: Existing `ModelChain.forward(*args, **kwargs)` routing and `Step.invoke()` results.
- Produces: Unchanged `ModelChain.forward()` signature; zero heads return only named stage outputs, one head returns a tensor, and multiple heads return a stage-keyed mapping.

- [ ] **Step 1: Change the headless-output test**

Replace `test_no_head_returns_the_merged_context` with a test named `test_no_head_returns_only_named_stage_outputs` asserting the exact keys `{"low", "high", "merged"}` and asserting that external input `"image"` is absent. Keep the existing value assertions in `test_a_step_may_produce_several_keys_at_once`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/model_chain/test_chain.py::test_no_head_returns_only_named_stage_outputs -q`

Expected: FAIL because the current result also contains `image`.

- [ ] **Step 3: Track produced values separately in `ModelChain.forward()`**

Keep the execution `context` for dependency routing, add a separate `produced: dict[str, object]`, update both mappings for non-head step results, and return `produced` when there are no terminal heads. Do not change one-head or multiple-head behavior.

- [ ] **Step 4: Run all ModelChain tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/model_chain -q`

Expected: PASS.

- [ ] **Step 5: Commit the isolated change**

```bash
git add src/geosave_engine/ml/model_chain/chain.py tests/ml/model_chain/test_chain.py
git diff --cached --check
git commit -m "fix: return only produced chain outputs"
```

### Task 2: Installed-code Transformers Adapter

**Files:**
- Modify: `src/geosave_engine/ml/huggingface.py`
- Rewrite tests in: `tests/ml/test_huggingface.py`

**Interfaces:**
- Consumes: `ModelChain.stage_specs`, `build_model(stages)`, and `geosave_engine.__about__.__version__`.
- Produces: `ARTIFACT_FORMAT_VERSION = 1`; `GeoSaveConfig(stages, format_version, geosave_version, **kwargs)`; strict `GeoSaveModel.from_chain(chain)` and `GeoSaveModel.from_pretrained(...)` behavior without remote-code registration.

- [ ] **Step 1: Write adapter-contract tests**

Use registered-name chains in adapter round-trip tests. Add assertions that:

```python
config["format_version"] == 1
config["geosave_version"] == __version__
"auto_map" not in config
not (release / "huggingface.py").exists()
```

Add `test_publication_rejects_class_path_stages`, `test_loading_rejects_unknown_format_version`, and `test_loading_rejects_different_geosave_version`. Preserve the exact missing/unexpected/mismatched-weight tests and same-process installed `AutoModel` registration coverage.

Rewrite the slow remote-code subprocess test as
`test_installed_adapter_loads_in_fresh_process`: the subprocess imports
`GeoSaveModel` from the installed package, loads the local artifact without
`trust_remote_code`, and compares exact output to the parent process.

- [ ] **Step 2: Run adapter tests and verify the new contract fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/test_huggingface.py -q`

Expected: FAIL because version fields and name-only publication do not exist and remote adapter code is still generated.

- [ ] **Step 3: Implement generated format and package versions**

In `GeoSaveConfig`, add typed `format_version: int = ARTIFACT_FORMAT_VERSION` and `geosave_version: str = __version__` arguments. Reject unsupported artifact versions with `ValueError` and different installed package versions with `RuntimeError`. Retain duplicate-stage validation.

- [ ] **Step 4: Enforce registered-name release recipes**

In `GeoSaveModel.from_chain()`, read the resolved recipe once and reject any stage whose selector does not contain a nonempty `name` or still contains `class_path`. Keep sharing the supplied modules and weights without rebuilding.

- [ ] **Step 5: Remove remote-code export registration**

Retain installed-process `AutoConfig.register()` and `AutoModel.register()`. Remove both `register_for_auto_class()` calls so `save_pretrained()` emits no `auto_map` or copied `huggingface.py`. Update adapter docstrings to describe it as internal persistence machinery.

- [ ] **Step 6: Run adapter tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/test_huggingface.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the isolated change**

```bash
git add src/geosave_engine/ml/huggingface.py tests/ml/test_huggingface.py
git diff --cached --check
git commit -m "refactor: keep model adapter code installed"
```

### Task 3: Local Release and Independent Loaders

**Files:**
- Create: `src/geosave_engine/release.py`
- Create: `tests/test_release.py`

**Interfaces:**
- Consumes: Task 2's `GeoSaveModel`, `ModelChain`, `ModelSpec.load/save`, `hf_hub_download()`, and the installed package version.
- Produces:
  - `save_model(model: ModelChain, path: str | Path, *, spec: ModelSpec | str | Path) -> Path`
  - `load_model(path_or_repo_id: str | Path, *, revision: str | None = None, token: str | bool | None = None, local_files_only: bool = False) -> ModelChain`
  - `load_spec(path_or_repo_id: str | Path, *, revision: str | None = None, token: str | bool | None = None, local_files_only: bool = False) -> ModelSpec`

- [ ] **Step 1: Write local release tests**

Build a lightweight registered `dense` head chain and a `ModelSpec` whose `model_inputs` produces `feature_map`. Add tests for:

- the exact four-file release layout;
- generated model-card text naming `geosave-engine[hub]==<version>`;
- `load_model(path)` returning a native `ModelChain` with identical output and state;
- `load_spec(path)` returning the matching validated `ModelSpec`;
- an existing destination raising `FileExistsError` without modification;
- a missing local `Path` making both loaders raise `FileNotFoundError` rather
  than calling Hub;
- a missing model input aborting before creating the target;
- an injected `save_pretrained()` failure leaving no target or staging directory.

- [ ] **Step 2: Write independent Hub-spec loader tests**

Patch `geosave_engine.release.hf_hub_download` and assert `load_spec("org/model", revision="abc", token="token", local_files_only=True)` requests only `ModelSpec.filename` with those arguments. Add a fresh-process test that loads a local spec and asserts `"transformers" not in sys.modules`.

- [ ] **Step 3: Run the new tests and verify imports fail**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/test_release.py -q`

Expected: collection FAIL because `geosave_engine.release` does not exist.

- [ ] **Step 4: Implement release validation and atomic local saving**

Create `geosave_engine.release` without importing Transformers at module import time. Add private helpers that:

- distinguish a `Path` as local even when missing, while an existing string path is local and any other string is a Hub ID;
- load and revalidate the supplied `ModelSpec`;
- require `set(model.inputs) <= set(spec.model_inputs)`;
- verify `model.stage_specs` with `json.dumps()` before writing;
- generate a minimal model card containing the exact package requirement and examples for `load_model()` and `load_spec()`.

`save_model()` must create a missing destination parent, stage in a sibling
temporary directory, call the internal adapter there, save the spec and card,
verify the four required files, then rename the directory to the absent target.
Cleanup on every failure and return the final target `Path`.

- [ ] **Step 5: Implement independent loaders**

`load_model()` imports `GeoSaveModel` inside the function, delegates local paths or repository IDs to `from_pretrained()`, and returns `.chain`. `load_spec()` delegates local sources to `ModelSpec.load()` and remote strings to `hf_hub_download(repo_id, filename=ModelSpec.filename, ...)` followed by `ModelSpec.load()`.

- [ ] **Step 6: Run local release tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/test_release.py tests/ml/test_huggingface.py -q`

Expected: PASS, with fresh-process tests marked slow and deselected by the project default.

- [ ] **Step 7: Run the fresh-process loader test**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -m slow tests/test_release.py tests/ml/test_huggingface.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the new public module**

```bash
git add src/geosave_engine/release.py tests/test_release.py
git diff --cached --check
git commit -m "feat: add native model release loaders"
```

### Task 4: Hugging Face Publication

**Files:**
- Modify: `src/geosave_engine/release.py`
- Modify: `tests/test_release.py`
- Remove obsolete upload coverage from: `tests/ml/test_huggingface.py`

**Interfaces:**
- Consumes: Task 3's `save_model()` and Hugging Face Hub `HfApi.create_repo()` / `HfApi.upload_folder()`.
- Produces: `publish_model(model: ModelChain, repo_id: str, *, spec: ModelSpec | str | Path, revision: str | None = None, token: str | bool | None = None) -> str` returning `CommitInfo.oid`.

- [ ] **Step 1: Write publication tests**

Mock `HfApi` and copy the uploaded folder into a test directory during the mock call. Assert:

```python
assert result == "immutable-commit-oid"
assert uploaded_names == {
    "README.md", "config.json", "model.safetensors", "model_spec.yaml"
}
```

Also assert `create_repo(..., repo_type="model", exist_ok=True)` and `upload_folder(..., revision="release-branch")` receive the supplied token. Reload the copied model and spec locally. Remove the old direct `GeoSaveModel.push_to_hub()` test.

- [ ] **Step 2: Run the publication test and verify it fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/test_release.py -k publish -q`

Expected: FAIL because `publish_model()` does not exist.

- [ ] **Step 3: Implement `publish_model()`**

Create a temporary parent directory, call `save_model()` into an absent child path, create the model repository, upload the complete folder, and return `CommitInfo.oid`. Do not call `PreTrainedModel.push_to_hub()` because it cannot include the independently owned spec atomically.

- [ ] **Step 4: Run all release and adapter tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/test_release.py tests/ml/test_huggingface.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the publication API**

```bash
git add src/geosave_engine/release.py tests/test_release.py tests/ml/test_huggingface.py
git diff --cached --check
git commit -m "feat: publish complete model releases"
```

### Task 5: Lightning Native-output Contract

**Files:**
- Modify: `src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py`
- Modify: `tests/ml/lightning/tasks/test_semantic_segmentation.py`

**Interfaces:**
- Consumes: Task 1's tensor-or-mapping `ModelChain.forward()` result.
- Produces: `SemanticSegmentationTask.forward(**model_inputs) -> dict[str, object] | torch.Tensor` that exactly delegates to `task.model`; private `_logits(result: object) -> torch.Tensor` used only by training lifecycle methods.

- [ ] **Step 1: Write native-output task tests**

Add a test model whose `@chain_step(outputs=("logits",))` returns logits without
declaring a terminal head. Assert the task and `task.model` results both have the
single key `"logits"`, then compare their tensors with
`torch.testing.assert_close()`. Add a training-step assertion proving the task
still extracts that mapping's logits for loss and metrics. Keep existing
tensor-head tests.

- [ ] **Step 2: Run the focused tests and verify forward fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/lightning/tasks/test_semantic_segmentation.py -k "forward or mapped" -q`

Expected: FAIL because `forward()` currently selects `result["logits"]`.

- [ ] **Step 3: Delegate `forward()` exactly and centralize logits selection**

Change `forward()` to return `self.model(**model_inputs)` unchanged. Add a concise private `_logits()` helper that accepts a tensor directly or selects a tensor under `"logits"` from a mapping, raising native `TypeError`/`KeyError` for invalid results. Use it in `training_step`, `validation_step`, `test_step`, and `predict_step`; keep raw logits as every lifecycle method's output.

- [ ] **Step 4: Run Lightning task and callback tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/ml/lightning/tasks/test_semantic_segmentation.py tests/ml/callbacks/test_task_callbacks.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the task contract**

```bash
git add src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py tests/ml/lightning/tasks/test_semantic_segmentation.py tests/ml/callbacks/test_task_callbacks.py
git diff --cached --check
git commit -m "refactor: expose native task model outputs"
```

### Task 6: Training Config Rename and User Documentation

**Files:**
- Rename: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model.yaml` to `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/train.yaml`
- Rename: `src/geosave_engine/templates/tasks/custom/lightning/configs/model.yaml` to `src/geosave_engine/templates/tasks/custom/lightning/configs/train.yaml`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md`
- Modify: `src/geosave_engine/ml/models/README.md`
- Modify: `tests/cli/core/test_workspace.py`

**Interfaces:**
- Consumes: Tasks 3-5 public release functions and Lightning behavior.
- Produces: Generated workspaces consistently use `configs/train.yaml`; documentation presents only `geosave_engine.release` and installed-code loading.

- [ ] **Step 1: Update workspace tests for the direct rename**

Change every generated-workspace assertion and CLI argument from `configs/model.yaml` to `configs/train.yaml`. Add assertions that `train.yaml` exists and `model.yaml` does not for both semantic-segmentation and custom-Lightning workspaces.

- [ ] **Step 2: Run workspace tests and verify they fail**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/cli/core/test_workspace.py -q`

Expected: FAIL because templates still contain `model.yaml`.

- [ ] **Step 3: Rename both template configs and update template instructions**

Rename the two YAML files without compatibility copies. Update the semantic template README commands and prose to use `train.yaml`; preserve `model_spec.yaml` as the independent workflow contract.

- [ ] **Step 4: Replace direct adapter documentation**

In `src/geosave_engine/ml/models/README.md`, replace `GeoSaveModel.from_chain()`, direct `save_pretrained()`, `push_to_hub()`, `AutoModel`, and `trust_remote_code` examples with `save_model()`, `publish_model()`, `load_model()`, and `load_spec()`. Include one backbone-only example and state that the repository contains no executable Python.

- [ ] **Step 5: Run workspace and release tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest tests/cli/core/test_workspace.py tests/test_release.py -q`

Expected: PASS.

- [ ] **Step 6: Commit templates and documentation**

```bash
git add src/geosave_engine/templates/tasks/semantic_segmentation/supervised src/geosave_engine/templates/tasks/custom/lightning/configs src/geosave_engine/ml/models/README.md tests/cli/core/test_workspace.py
git diff --cached --check
git commit -m "docs: adopt native model release workflow"
```

### Task 7: Integrated Verification

**Files:**
- Verify only; modify the owning task's files if a failure exposes a regression.

**Interfaces:**
- Consumes: All preceding tasks.
- Produces: Evidence that release, Lightning, templates, and dense prediction agree on tensor and mapping outputs.

- [ ] **Step 1: Run the focused behavior suite**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest \
  tests/ml/model_chain \
  tests/ml/test_huggingface.py \
  tests/test_release.py \
  tests/ml/lightning/tasks/test_semantic_segmentation.py \
  tests/ml/callbacks/test_task_callbacks.py \
  tests/cli/core/test_workspace.py \
  tests/workflow/test_prediction.py -q
```

Expected: PASS.

- [ ] **Step 2: Run fresh-process release verification**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -m slow tests/test_release.py tests/ml/test_huggingface.py -q`

Expected: PASS.

- [ ] **Step 3: Run scoped lint and type checks**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check \
  src/geosave_engine/release.py \
  src/geosave_engine/ml/huggingface.py \
  src/geosave_engine/ml/model_chain/chain.py \
  src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py \
  tests/test_release.py \
  tests/ml/test_huggingface.py \
  tests/ml/model_chain/test_chain.py \
  tests/ml/lightning/tasks/test_semantic_segmentation.py \
  tests/cli/core/test_workspace.py

UV_CACHE_DIR=/tmp/geosave-uv-cache uv run basedpyright \
  src/geosave_engine/release.py \
  src/geosave_engine/ml/huggingface.py \
  src/geosave_engine/ml/model_chain/chain.py \
  src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py
```

Expected: both commands exit 0.

- [ ] **Step 4: Inspect the final patch**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; all unrelated pre-existing changes remain untouched.

- [ ] **Step 5: Commit only verification fixes, if any**

If verification required product changes, stage only those task-owned hunks, inspect `git diff --cached`, and commit them as `fix: complete native model release verification`. Otherwise do not create an empty commit.
