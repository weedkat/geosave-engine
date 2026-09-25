# Workflow Call Execution Design

## Purpose

Make a model-spec call a complete, reusable unit of parsing and execution.
Preprocessing becomes a Prefect flow that creates one independently monitored task
run for each declared call. The design removes the stage-wide `Preprocessor` and
the shallow `run_call` adapter while preserving explicit references, literal
keyword arguments, lazy xarray values, and inert model-spec loading.

This document supersedes the call and preprocessing-execution sections of
`2026-09-25-workflow-redesign.md`. Source requirements, deployment configs,
ingestion, persistence, tensor conversion, and the postprocessing skeleton keep
their existing contracts unless this document explicitly changes them.

## Structure

```text
workflow/
├── specs/
│   ├── call.py          # Ref, CallValue, CallSpec, CallStage
│   └── model.py         # ModelSpec composes CallStage
├── flows/
│   ├── ingest.py       # deployable raster ingestion
│   └── preprocess.py   # one task run per CallSpec
└── tasks/
    ├── load.py         # one source load
    └── save.py         # one completed stack write
```

Remove `specs/preprocessing.py` and `tasks/preprocess.py`. There is no
`tasks/call.py`, `run_call`, or stage-wide execution class.

## Call specification

`specs/call.py` owns the complete model-spec grammar for calls:

```python
type CallValue = (
    None
    | bool
    | int
    | float
    | str
    | Ref
    | list[CallValue]
    | dict[str, CallValue]
)


class CallSpec(SpecModel):
    call: str | Ref
    kwargs: dict[str, CallValue] = Field(default_factory=dict)

    @property
    def references(self) -> tuple[Ref, ...]: ...

    @property
    def inputs(self) -> frozenset[str]: ...

    def bindings(self, values: Mapping[str, Any]) -> dict[str, Any]: ...

    def invoke(self, values: Mapping[str, Any]) -> Any: ...
```

`call` remains the YAML field. `invoke` is the operation because a method named
`call` would collide with that field.

`CallSpec` owns all of the following behavior:

- validate an imported module-level path or an explicit `Ref`;
- validate nested finite YAML literals and references in `kwargs`;
- enumerate referenced root names in encounter order;
- select only the root bindings needed by this call;
- resolve the target and nested arguments against supplied bindings;
- validate the resolved callable signature before invocation; and
- invoke exactly once and return the complete result, including `None`.

Plain strings remain literals. Only `!ref` values are references. Parsing and
model-spec loading do not import or invoke declared targets. Imported paths and
bound methods are resolved when `invoke` runs.

Literal containers are rebuilt for every invocation so mutating callables cannot
change the stored specification. Exceptions keep their native type and receive
one note identifying the declared call path.

## Ordered call stage

`CallStage` is a Pydantic root model over an ordered
`dict[Name, CallSpec]`. Both `ModelSpec.preprocessing` and
`ModelSpec.inference` use it, so parsing, dependency validation, and YAML
round-tripping have one implementation.

It implements `Mapping[str, CallSpec]`; callers retain ordinary indexing,
iteration, membership, and `.items()` without reaching into a Pydantic `root`
attribute. `model_dump()` and YAML persistence still emit the existing plain
mapping shape.

The stage derives its external inputs before execution. A reference is valid
when it names:

- a value supplied to the stage;
- an earlier call result; or
- the current output name when an external value with that name is being
  deliberately rebound.

A reference to a later output is a forward reference and fails before any task
is submitted. A missing external input also fails before submission. Independent
calls may share inputs and do not gain false ordering dependencies.

`CallStage` does not import call targets. A missing package, non-callable bound
attribute, or incompatible signature belongs to the individual task run that
attempted that call.

## Preprocessing flow

`flows/preprocess.py` exposes a Prefect `preprocess` flow:

```python
preprocess(
    values: Mapping[str, Any],
    spec: ModelSpec,
) -> dict[str, Any]
```

The flow performs these steps in order:

1. Revalidate the model specification and copy the supplied mapping.
2. Validate every external and forward reference through `CallStage`.
3. Select and validate only model sources consumed as external stage inputs.
4. For each declaration, take only `CallSpec.bindings(state)`.
5. Adapt `CallSpec.invoke` directly into a named Prefect task and submit it.
6. Bind its future under the declaration name for downstream dependencies.
7. Resolve submitted futures only after the graph has been constructed.
8. Return supplied values, selected sources, and completed named results.

Task construction is direct:

```python
operation = task(
    name=f"preprocessing.{name}",
    cache_policy=NO_CACHE,
    persist_result=False,
)(declaration.invoke)
```

There is no wrapper function. Each YAML entry produces its own task run and
therefore its own state, duration, failure, logs, and future retry configuration.
Prefect recursively resolves futures inside a call's binding mapping, so
references to earlier results form explicit dependency edges. Independent calls
can run concurrently under Prefect's default thread-pool task runner.

The flow and its call tasks disable result persistence. Native lazy xarray
objects remain in process; completed files remain the durable workflow seam.

Calling `CallSpec.invoke` directly remains the ordinary-Python execution seam
for one declaration. The library does not keep a second stage executor alongside
the Prefect flow.

## Inference

Inference continues to be parsed and validated as a `CallStage` but is not
executed. Its declarations therefore gain the same call grammar and dependency
checks without introducing sampling, batching, model loading, prediction,
tiling, merging, or postprocessing behavior.

## Prefect consistency

All Prefect tasks use `NO_CACHE` rather than the runtime-supported but
incorrectly typed `cache_policy=None`. The ingest flow uses Prefect's default
`ThreadPoolTaskRunner` rather than constructing the same runner explicitly.
These choices preserve behavior and agree with Prefect's public type interface.

Decorated functions remain typed as Prefect `Task` or `Flow` objects, so `.fn`,
`.submit`, `cache_policy`, and `persist_result` are visible to basedpyright.

## Type-checking cleanup

`pyrightconfig.json` explicitly binds the repository `.venv`; without that,
basedpyright omits installed packages and reports false missing imports.

The implementation also resolves genuine workflow diagnostics rather than
suppressing them:

- narrow optional source selectors and STAC identities after validation;
- give the YAML `!ref` constructor a concrete scalar-node signature;
- validate raw Pydantic input with `model_validate` where constructor typing
  intentionally does not model coercion;
- narrow path validator inputs before `Path` construction;
- type test fixtures against the protocols they satisfy; and
- use public imports and correctly typed HTTP handler overrides.

The acceptance target is zero Ruff and basedpyright diagnostics for
`src/geosave_engine/workflow` and `tests/workflow`, without per-line ignores.
Unrelated in-progress ML migrations remain outside this work.

## Errors and monitoring

- Invalid YAML shape, references, dependency order, or missing stage inputs fail
  before task submission.
- Source incompatibility fails before task submission and before pixel work.
- Import, attribute, callability, signature, and callable-body errors fail the
  named task that owns the declaration.
- Later tasks depending on a failed result become `NotReady`; independent calls
  may still complete according to Prefect semantics.
- Model-spec loading remains safe and inert even when an inference-only import
  is unavailable.

## Testing

Tests cover the public behavior rather than implementation structure:

- `CallSpec` validates and invokes imported calls, bound methods, nested
  references, literals, rebinding, `None` results, and fresh containers.
- `CallStage` rejects missing and forward references before any callable runs.
- A preprocessing flow creates distinct named task runs and preserves dependency
  order while allowing independent calls to overlap.
- Each call receives only its referenced roots.
- Source selection remains lazy and occurs before task submission.
- An inactive inference import remains inert.
- Task results are neither cached nor persisted.
- Existing model-spec YAML round trips unchanged apart from the Python type name
  `OperationSpec` becoming `CallSpec`.
- Ruff and basedpyright pass for workflow source and tests.

The real Prefect integration test uses the temporary local server. Restricted
sandboxes may require loopback permission, as with existing workflow tests.

## Breaking changes

- `OperationSpec` is renamed to `CallSpec` with no compatibility alias.
- `ModelSpec.preprocessing` and `ModelSpec.inference` become `CallStage` values
  that retain mapping behavior and their existing YAML representation.
- `Preprocessor` and the `preprocess` Prefect task are removed.
- `preprocess` becomes a Prefect flow with one task run per declaration.
- Invalid imported calls fail in their named task rather than while constructing
  a stage-wide executor.
