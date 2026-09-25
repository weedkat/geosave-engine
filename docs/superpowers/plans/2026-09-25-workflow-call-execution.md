# Workflow Call Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stage-wide preprocessing executor with model-owned call specifications and one decorated Prefect task run per declared preprocessing call.

**Architecture:** `CallSpec` owns parsing and invoking one inert declaration, while `StageSpec` owns the ordered declaration mapping and dependency validation. A stable `invoke_call` Prefect task adapts the pure call method, and the `preprocess` flow validates inputs, submits named task runs, and returns only declared outputs.

**Tech Stack:** Python 3.12, Pydantic 2, PyYAML, xarray/Dask, Prefect 3, pytest, Ruff, basedpyright.

**Spec:** `docs/superpowers/specs/2026-09-25-workflow-call-execution-design.md`

## Global Constraints

- Preserve unrelated working-tree changes and stage only files listed by the active task.
- Do not inspect or modify notebooks.
- Do not add compatibility aliases, registries, expression languages, inferred string references, or a second stage executor.
- `model_spec.yaml` remains the sole public workflow YAML artifact.
- Plain strings are literals; only `!ref` values are runtime references.
- `specs` and `configs` import neither Prefect nor Lightning.
- Use one stable decorated `invoke_call` task; each declaration becomes a named Prefect `TaskRun`, not a dynamically created task definition.
- Preprocessing preserves native objects and Dask laziness and returns only declared stage results; an empty stage returns `{}`.
- Inference is parsed as a `StageSpec` but is not executed.
- Postprocessing, sampling, tiling, merging, model loading, and prediction remain unimplemented.
- Use Prefect's default task runner, `NO_CACHE`, and `persist_result=False` for native in-process results.
- Use `UV_CACHE_DIR=/tmp/geosave-call-execution-cache` if the default uv cache is unwritable.

## Review Focus

- Rebinding an output name must read the supplied value before replacing it; Task 1 tests this.
- A later-stage reference must fail before submission unless that name was supplied explicitly; Tasks 1 and 2 test this.
- Inputs named `wait_for` and `return_state` must not collide with Prefect keywords; Tasks 1 and 2 pass one positional mapping and test both.
- Mutating callables must receive fresh literal containers on every invocation; Task 1 tests this.
- Missing imports, non-callable targets, bad signatures, and callable exceptions must fail the owning task run while model loading stays inert; Tasks 1 and 2 test this.

---

### Task 1: Add pure call and stage specifications

**Files:**
- Create: `src/geosave_engine/workflow/specs/call.py`
- Create: `src/geosave_engine/workflow/specs/stage.py`
- Modify: `src/geosave_engine/workflow/specs/preprocessing.py`
- Modify: `src/geosave_engine/workflow/specs/__init__.py`
- Create: `tests/workflow/specs/test_call.py`
- Create: `tests/workflow/specs/test_stage.py`

**Interfaces:**
- Consumes: `SpecModel`, `Name`, and the existing `!ref` representation.
- Produces: `Ref`, recursive `CallValue`, `CallSpec.select_inputs(state)`, `CallSpec.invoke(inputs, /)`, and mapping-like `StageSpec.validate_inputs(names)`.

- [ ] **Step 1: Write failing `CallSpec` tests**

Create module-level callables and tests in `tests/workflow/specs/test_call.py`:

```python
def mutate(*, items: list[int]) -> list[int]:
    items.append(2)
    return items


def test_call_invokes_imports_bound_methods_and_nested_references() -> None:
    imported = CallSpec.model_validate({
        "call": "builtins.dict",
        "kwargs": {
            "nested": [Ref("source")],
            "literal": {"ref": "source"},
            "text": "!ref source",
        },
    })
    source = object()
    assert imported.invoke({"source": source}) == {
        "nested": [source],
        "literal": {"ref": "source"},
        "text": "!ref source",
    }
    assert CallSpec(call=Ref("number.as_integer_ratio")).invoke(
        {"number": 1.5}
    ) == (3, 2)


def test_call_selects_only_referenced_roots() -> None:
    spec = CallSpec(
        call=Ref("function"),
        kwargs={"left": Ref("wait_for"), "right": Ref("return_state")},
    )
    selected = spec.select_inputs({
        "function": lambda **kwargs: kwargs,
        "wait_for": 1,
        "return_state": 2,
        "unused": object(),
    })
    assert set(selected) == {"function", "wait_for", "return_state"}


def test_call_rebuilds_literal_containers() -> None:
    spec = CallSpec(call=Ref("mutate"), kwargs={"items": [1]})
    first = spec.invoke({"mutate": mutate})
    second = spec.invoke({"mutate": mutate})
    assert first == second == [1, 2]
    assert first is not second
```

Also test invalid reference paths, non-finite floats, non-string keys, cyclic containers, `None` results, missing attributes, non-callable targets, signature failure before invocation, deferred missing imports, and an exception note containing the declared call path.

- [ ] **Step 2: Write failing `StageSpec` tests**

Create `tests/workflow/specs/test_stage.py`:

```python
def stage(**calls: object) -> StageSpec:
    return StageSpec.model_validate(calls)


def test_stage_is_an_ordered_mapping_with_plain_dump_shape() -> None:
    spec = stage(first={"call": "builtins.dict"},
                 second={"call": "builtins.list"})
    assert list(spec) == ["first", "second"]
    assert spec["first"].call == "builtins.dict"
    assert spec.model_dump() == {
        "first": {"call": "builtins.dict", "kwargs": {}},
        "second": {"call": "builtins.list", "kwargs": {}},
    }


def test_stage_validates_external_inputs_and_order() -> None:
    spec = stage(
        selected={"call": Ref("source.select")},
        result={"call": "builtins.dict", "kwargs": {"value": Ref("selected")}},
    )
    assert spec.validate_inputs({"source", "unused"}) == frozenset({"source"})


def test_stage_rejects_missing_and_forward_references() -> None:
    missing = stage(result={"call": Ref("source.method")})
    forward = stage(first={"call": Ref("second")},
                    second={"call": "builtins.dict"})
    with pytest.raises(ValueError, match="source"):
        missing.validate_inputs(set())
    with pytest.raises(ValueError, match="Forward.*second"):
        forward.validate_inputs(set())
    assert forward.validate_inputs({"second"}) == frozenset({"second"})
```

Add a rebinding case where `value` is both supplied and the current output name.

- [ ] **Step 3: Run the new tests and verify collection fails**

```bash
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run pytest \
  tests/workflow/specs/test_call.py tests/workflow/specs/test_stage.py -q
```

Expected: imports fail because `CallSpec` and `StageSpec` do not exist.

- [ ] **Step 4: Implement `Ref` and `CallSpec`**

Move `Ref` into `specs/call.py` and implement:

```python
type CallValue = (
    None | bool | int | float | str | Ref
    | list[CallValue] | dict[str, CallValue]
)


class CallSpec(SpecModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    call: str | Ref
    kwargs: dict[str, CallValue] = Field(default_factory=dict)

    @property
    def references(self) -> tuple[Ref, ...]:
        return self._find_references([self.call, self.kwargs])

    @property
    def inputs(self) -> frozenset[str]:
        return frozenset(reference.root for reference in self.references)

    def select_inputs(self, state: Mapping[str, object]) -> dict[str, object]:
        if missing := self.inputs - state.keys():
            raise ValueError(f"Missing call inputs: {sorted(missing)}")
        return {
            reference.root: state[reference.root]
            for reference in self.references
        }

    def invoke(self, inputs: Mapping[str, object], /) -> object:
        try:
            target = self._resolve_target(inputs)
            kwargs = self._resolve_value(self.kwargs, inputs)
            self._bind(target, kwargs)
            return target(**kwargs)
        except Exception as error:
            error.add_note(f"While invoking {self.call!r}")
            raise
```

Implement `_find_references` as the recursive tuple walk already tested on `OperationSpec`; implement `_resolve_value` as the corresponding recursive fresh-container rebuild. `_resolve_target` imports a string with `import_module` and resolves a `Ref` against `inputs`; it rejects non-callables. `_bind` uses `inspect.signature(target).bind(**kwargs)` and skips only targets whose signatures Python cannot inspect. `select_inputs` rejects missing roots and omits unrelated state. `invoke` adds one note identifying `self.call` and re-raises the native exception.

During this task only, import `Ref` into `specs/preprocessing.py` so existing `OperationSpec` tests remain valid until Task 2 deletes that file. Do not create an alias between `OperationSpec` and `CallSpec`.

- [ ] **Step 5: Implement `StageSpec`**

Use a root model implementing `Mapping[str, CallSpec]`:

```python
class StageSpec(RootModel[dict[Name, CallSpec]], Mapping[str, CallSpec]):
    root: dict[Name, CallSpec] = Field(default_factory=dict)

    def __getitem__(self, name: str) -> CallSpec:
        return self.root[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    @property
    def external_inputs(self) -> tuple[str, ...]:
        assigned: set[str] = set()
        external: dict[str, None] = {}
        for output, call in self.root.items():
            for reference in call.references:
                if reference.root not in assigned:
                    external.setdefault(reference.root, None)
            assigned.add(output)
        return tuple(external)

    def validate_inputs(self, names: Collection[str]) -> frozenset[str]:
        supplied = set(names)
        declared = set(self.root)
        assigned: set[str] = set()
        required: set[str] = set()
        for output, call in self.root.items():
            for name in call.inputs:
                if name in assigned:
                    continue
                if name in supplied:
                    required.add(name)
                elif name in declared:
                    raise ValueError(f"Forward stage reference: {name!r}")
                else:
                    required.add(name)
            assigned.add(output)
        if missing := required - supplied:
            raise ValueError(f"Missing stage inputs: {sorted(missing)}")
        return frozenset(required)
```

`validate_inputs` walks declarations in order. Earlier outputs satisfy later references; supplied names satisfy external references and deliberate rebinding; an unsupplied later output raises `Forward stage reference`; all other absent roots raise `Missing stage inputs`. Export the new types from `specs/__init__.py`.

- [ ] **Step 6: Verify and commit the specification layer**

```bash
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run pytest \
  tests/workflow/specs/test_call.py tests/workflow/specs/test_stage.py \
  tests/workflow/specs/test_model.py -q
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run ruff check \
  src/geosave_engine/workflow/specs tests/workflow/specs
git add src/geosave_engine/workflow/specs tests/workflow/specs/test_call.py \
  tests/workflow/specs/test_stage.py
git commit -m "refactor: define workflow call stages"
```

Expected: focused tests and Ruff pass.

### Task 2: Execute preprocessing as Prefect task runs

**Files:**
- Modify: `src/geosave_engine/workflow/specs/model.py`
- Modify: `src/geosave_engine/workflow/specs/__init__.py`
- Delete: `src/geosave_engine/workflow/specs/preprocessing.py`
- Create: `src/geosave_engine/workflow/tasks/call.py`
- Modify: `src/geosave_engine/workflow/tasks/__init__.py`
- Delete: `src/geosave_engine/workflow/tasks/preprocess.py`
- Create: `src/geosave_engine/workflow/flows/preprocess.py`
- Modify: `src/geosave_engine/workflow/flows/__init__.py`
- Modify: `tests/workflow/specs/test_model.py`
- Create: `tests/workflow/tasks/test_call.py`
- Delete: `tests/workflow/tasks/test_preprocess.py`
- Create: `tests/workflow/flows/test_preprocess.py`

**Interfaces:**
- Consumes: Task 1's `CallSpec` and `StageSpec` APIs.
- Produces: `ModelSpec.preprocessing: StageSpec`, `ModelSpec.inference: StageSpec`, decorated `invoke_call`, and `preprocess(inputs, spec) -> dict[str, object]`.

- [ ] **Step 1: Change model and task tests first**

Replace `OperationSpec` uses in `test_model.py` with `CallSpec`, assert both stages are `StageSpec`, and assert `model_dump()` keeps the plain mapping shape. Create `tests/workflow/tasks/test_call.py`:

```python
def test_invoke_call_is_one_nonpersisted_uncached_task() -> None:
    assert isinstance(invoke_call, Task)
    assert invoke_call.name == "model-call"
    assert invoke_call.cache_policy is NO_CACHE
    assert invoke_call.persist_result is False


def test_invoke_call_delegates_to_the_spec() -> None:
    spec = CallSpec(
        call="builtins.pow",
        kwargs={"base": Ref("value"), "exp": 2},
    )
    assert invoke_call.fn(spec, {"value": 3}) == 9
```

Add a task test passing roots named `wait_for` and `return_state` inside the positional input mapping.

- [ ] **Step 2: Write preprocessing flow tests**

Create `tests/workflow/flows/test_preprocess.py` with:

```python
def test_empty_preprocessing_returns_no_supplied_values(prefect_server) -> None:
    assert preprocess({"source": object()}, model_spec()) == {}


def test_preprocessing_returns_only_declared_results(prefect_server) -> None:
    spec = model_spec(preprocessing={
        "squared": {
            "call": "builtins.pow",
            "kwargs": {"base": Ref("value"), "exp": 2},
        },
        "ratio": {"call": Ref("squared.as_integer_ratio")},
    })
    assert preprocess({"value": 3, "unused": object()}, spec) == {
        "squared": 9,
        "ratio": (9, 1),
    }
```

Migrate the durable behaviors from `test_preprocess.py`: validate every reference and consumed source before submission, preserve lazy xarray values, select only consumed sources, rebind from the old value, preserve `None`, keep inference imports inert, enforce dependency order, allow independent calls to enter a shared `threading.Barrier(2)`, and expose `preprocess-{name}` in captured Prefect logs. Validation tests call `preprocess.fn` because they fail before `.submit()`; successful tests use `prefect_server`.

- [ ] **Step 3: Run changed tests and verify missing interfaces**

```bash
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run pytest \
  tests/workflow/specs/test_model.py tests/workflow/tasks/test_call.py \
  tests/workflow/flows/test_preprocess.py -q
```

Expected: imports fail and `ModelSpec` still exposes `OperationSpec` dictionaries.

- [ ] **Step 4: Make `ModelSpec` compose stages**

Use:

```python
preprocessing: StageSpec = Field(default_factory=StageSpec)
inference: StageSpec = Field(default_factory=StageSpec)
```

Keep YAML loading inert and round-trip stages as plain mappings. Replace the lambda YAML constructor with a named function accepting `yaml.ScalarNode`. Remove `OperationSpec` exports and delete `specs/preprocessing.py`.

- [ ] **Step 5: Add the decorated task**

Create `tasks/call.py`:

```python
@task(name="model-call", cache_policy=NO_CACHE, persist_result=False)
def invoke_call(
    spec: CallSpec,
    inputs: Mapping[str, object],
    /,
) -> object:
    """Invoke one model-owned call as an observable task run."""
    return spec.invoke(inputs)
```

Export it from `tasks/__init__.py`. Do not decorate `CallSpec.invoke` or construct tasks dynamically.

- [ ] **Step 6: Add the preprocessing flow**

Create `flows/preprocess.py` with this sequence:

```python
model = spec.validated_copy()
stage = model.preprocessing
required = stage.validate_inputs(inputs.keys())
state = dict(inputs)

for name in stage.external_inputs:
    if name in required and name in model.sources:
        state[name] = model.sources[name].select_raster(state[name])

futures = {}
for name, declaration in stage.items():
    operation = invoke_call.with_options(task_run_name=f"preprocess-{name}")
    future = operation.submit(declaration, declaration.select_inputs(state))
    state[name] = future
    futures[name] = future

return {name: future.result() for name, future in futures.items()}
```

Define `StageSpec.external_inputs` in Task 1 as an encounter-ordered tuple so source selection is deterministic. Wrap source errors with `Source {name!r}` before submission. Export `preprocess` from `flows/__init__.py`, remove `preprocess` from `tasks/__init__.py`, and delete the old task and class.

- [ ] **Step 7: Verify and commit the execution boundary**

```bash
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run pytest \
  tests/workflow/specs/test_call.py tests/workflow/specs/test_stage.py \
  tests/workflow/specs/test_model.py tests/workflow/tasks/test_call.py \
  tests/workflow/flows/test_preprocess.py -q
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run ruff check \
  src/geosave_engine/workflow tests/workflow/specs \
  tests/workflow/tasks/test_call.py tests/workflow/flows/test_preprocess.py
git add src/geosave_engine/workflow/specs src/geosave_engine/workflow/tasks \
  src/geosave_engine/workflow/flows tests/workflow/specs/test_model.py \
  tests/workflow/tasks tests/workflow/flows/test_preprocess.py
git commit -m "refactor: execute preprocessing as task runs"
```

Expected: focused tests and Ruff pass. A telemetry SQLite-lock log is acceptable only when flow and task states still complete.

### Task 3: Migrate shipped consumers

**Files:**
- Modify: `tests/workflow/specs/test_examples.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md`

**Interfaces:**
- Consumes: `ModelSpec.load(path)` and `preprocess(inputs, spec)`.
- Produces: template code with no `Preprocessor` or `workflow.tasks.preprocess` imports.

- [ ] **Step 1: Change the example tests before template code**

Replace executor use with:

```python
prepared = preprocess(
    {"sentinel_2_l2a": optical},
    ModelSpec.load(path),
)
assert set(prepared) == {"valid_pixels", "image"}
assert isinstance(prepared["image"].B04.data, da.Array)
```

Add `prefect_server` to successful flow tests. Preserve lazy-pixel, literal, rebinding, `None`, and YAML round-trip assertions.

- [ ] **Step 2: Verify the obsolete consumer fails**

```bash
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run pytest \
  tests/workflow/specs/test_examples.py -q
```

Expected: collection fails because `Preprocessor` was removed.

- [ ] **Step 3: Update the workspace source and README**

Use this public example in both places:

```python
from geosave_engine.workflow.flows import preprocess
from geosave_engine.workflow.specs import ModelSpec

spec = ModelSpec.load("configs/model_spec.yaml")
image = preprocess({"sentinel_2_l2a": optical}, spec)["image"]
```

Keep the README focused on input, output, and what is not implemented. Do not document internal reference walking or task construction.

- [ ] **Step 4: Verify and commit the consumers**

```bash
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run pytest \
  tests/workflow/specs/test_examples.py tests/cli/core/test_workspace.py -q
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run ruff check \
  tests/workflow/specs/test_examples.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py
git add tests/workflow/specs/test_examples.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md
git commit -m "docs: use preprocessing flow in workspace"
```

Expected: example, workspace smoke, and Ruff checks pass.

### Task 4: Make the workflow scope lint- and type-clean

**Files:**
- Modify: `pyrightconfig.json`
- Modify: `src/geosave_engine/workflow/configs/anchor.py`
- Modify: `src/geosave_engine/workflow/flows/ingest.py`
- Modify: `src/geosave_engine/workflow/specs/model.py`
- Modify: `src/geosave_engine/workflow/specs/sources.py`
- Modify: `src/geosave_engine/workflow/tasks/load.py`
- Modify: `src/geosave_engine/workflow/tasks/save.py`
- Modify: `tests/workflow/conftest.py`
- Modify: `tests/workflow/flows/test_ingest.py`
- Modify: `tests/workflow/specs/test_model.py`
- Modify: `tests/workflow/specs/test_sources.py`
- Modify: `tests/workflow/tasks/test_load.py`
- Modify: `tests/workflow/tasks/test_save.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: zero Ruff and basedpyright diagnostics for workflow source and tests.

- [ ] **Step 1: Bind basedpyright to the repository environment**

Add to `pyrightconfig.json`:

```json
"venvPath": ".",
"venv": ".venv",
```

Run `UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run basedpyright src/geosave_engine/workflow tests/workflow` and retain its output as the failing baseline.

- [ ] **Step 2: Use Prefect's typed public configuration**

Use `NO_CACHE` instead of `cache_policy=None` in `tasks/load.py` and `tasks/save.py`. Remove the explicit `ThreadPoolTaskRunner(max_workers=4)` from `flows/ingest.py`. Update assertions to use `task.cache_policy is NO_CACHE` and keep `persist_result is False`.

- [ ] **Step 3: Narrow validated source and path values**

After checking `RasterRequirement.collection` and `.endpoints`, store them on `RasterLoader` as non-optional `self.collection` and `self.endpoints` and use those attributes. In `_select_variables` and `_select_channels`, narrow `variables` or `channels` to a local non-optional value before `set`, `list`, comparisons, and slicing. In `GeoJSONAnchorConfig.validate_path`, reject values that are neither `str` nor `os.PathLike` before `Path(value)`.

- [ ] **Step 4: Make tests type their coercion boundaries explicitly**

Use `RasterRequirement.model_validate`, `ModelSpec.model_validate`, and `SourceConfig.model_validate` when tests intentionally supply URL strings or nested dictionaries. Import `delayed` from `dask.delayed`. Give HTTP-handler overrides the base signatures, narrow the server to `ThreadingHTTPServer`, and replace `StacSource(client=None, ...)` with a small fixture double satisfying `SearchClient`. Do not add `type: ignore` comments.

- [ ] **Step 5: Run and fix the complete owned scope**

```bash
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run basedpyright \
  src/geosave_engine/workflow tests/workflow
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run ruff check \
  src/geosave_engine/workflow tests/workflow
UV_CACHE_DIR=/tmp/geosave-call-execution-cache uv run pytest tests/workflow -q
```

Expected: basedpyright reports `0 errors, 0 warnings, 0 notes`; Ruff and all workflow tests pass.

- [ ] **Step 6: Audit removed surfaces and the final diff**

```bash
rg -n "OperationSpec|Preprocessor|tasks\.preprocess|cache_policy=None|ThreadPoolTaskRunner|\bacquire\b" \
  src/geosave_engine/workflow tests/workflow \
  src/geosave_engine/templates --glob '!*.ipynb'
git diff --check
git status --short
```

Expected: no obsolete execution API, incorrect cache policy, explicit default runner, forbidden workflow name, or whitespace error remains. Unrelated ML/geodata changes remain unstaged.

- [ ] **Step 7: Commit the quality pass**

```bash
git add pyrightconfig.json src/geosave_engine/workflow tests/workflow
git commit -m "fix: make workflow checks type clean"
```

Stage only the workflow files named in this task; do not stage unrelated pre-existing test migrations.
