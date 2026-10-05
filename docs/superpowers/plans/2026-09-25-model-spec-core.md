# Model spec core implementation plan

**Goal:** Implement the approved call/reference structure and independent processors; leave inference execution and model loading undecided.

**Architecture:** `ml.spec` owns the version-2 document and inert references. `ml.processing` executes only the selected stage using native objects. Move existing requirement validators into this independent package and update their current consumers; do not duplicate them or adapt version-1 inference into version 2.

**Tech stack:** Existing Python, Pydantic, PyYAML, xarray and pytest dependencies.

**Spec:** [Core design](../specs/2026-09-25-workflow-core-design.md), with the user's explicit restriction that inference execution/model loading are deferred.

## Scope and decisions

- Work in the shared checkout; preserve existing draft documents and unrelated files.
- Python round-trip means typed configuration objects/dictionaries and YAML, not conversion of arbitrary Python programs into YAML.
- Preserve existing workflow execution until its inference replacement is agreed. No compatibility dispatch or v1-to-v2 conversion.
- Retain `OperationSpec`, `ModelSpec`, `call`, `kwargs`, `!ref` and native geodata names.
- The new document requires explicit version 2; parsing never imports or executes call targets.
- Raster requirements keep their existing defaults, with the source kind defaulting to `raster`.
- Do not implement new inference helpers, model clients, serving integration or code generation.

## Task 1: Portable declarations

Files: `ml/spec/{base,requirements,references,model,__init__}.py`; existing workflow imports; `tests/ml/spec/test_model.py`.

- [x] Write and run failing tests for `Ref`, nested references/literals, imported/bound calls, typed/Python/YAML round trips, duplicate keys, invalid paths, unknown fields/versions and inactive missing imports.
- [x] Move existing validation owners; add inert reference parsing/dumping and version-2 `ModelSpec`/`OperationSpec`.
- [x] Verify new tests and existing workflow requirement/schema tests.

Public usage:

```python
spec = ModelSpec(schema_version=2, sources={}, preprocessing={
    'value': OperationSpec(call='builtins.dict', kwargs={'value': Ref('input')})
})
assert ModelSpec.load(spec.save(directory)) == spec
assert ModelSpec.model_validate(spec.model_dump()) == spec
```

## Task 2: Independent execution

Files: `ml/processing.py`, `tests/ml/test_processing.py`.

- [x] Write and run failing tests for lazy native data, active-source validation, method binding, supplied callables, recursive kwargs, rebinding/aliasing, actual None returns and missing-reference validation before work.
- [x] Implement `Processor(spec, stage="preprocessing")`, `Processor(spec, stage="postprocessing")`, `.load(path)` and calls taking a supplied-value mapping and returning a fresh mapping with stage results.
- [x] Validate active imports/signatures without loading inactive functions; resolve references in document order without implicit conversion or input mutation.
- [x] Verify repeated calls and fresh-process independence from Prefect/Lightning.

```python
prepare = Processor.load(path, stage="preprocessing")
prepared = prepare({'optical': optical})
finish = Processor.load(path, stage="postprocessing")
results = finish({'logits': logits, 'mask': mask})
```

## Task 3: Native sequence boundary and documentation

Files: `geodata/transform/tiling.py`, `tests/geodata/transform/test_tiling.py`, `ml/spec/README.md`, approved draft status sections.

- [x] Reproduce YAML-list tile_shape failure with a regression test.
- [x] Normalize the sequence once in `Tiles`; retain accumulator lifecycle.
- [x] Document executable core usage, source preservation, explicit side effects and round-trip limits; mark inference proposals unapproved/deferred.
- [x] Run affected tests, lint and diff checks; review the final patch.

## Review focus

- Inactive inference imports must never block loading preprocessing.
- Resolved Python objects must not be traversed as configuration.
- Repeated calls must not reuse mutated kwargs containers or binding dictionaries.
- Required source validation must preserve extra bands, native object identity and Dask laziness.
- YAML aliases/duplicate keys/non-string or tagged keys must not bypass validation.

## Initial execution record (superseded by the cleanup below)

User authorized implementation of the structure, explicitly excluding agreement on inference. Implementation proceeds in this session with tests before production edits.

Validation: 179 schema/processing/tiling tests and 81 existing workflow integration
tests passed (260 total); affected-file Ruff and whitespace checks passed. New
features and the sequence correction were first observed failing before their
implementation. Independent review found no blocking issues. Reviewer noted that
direct mutation of private `Ref._path` can invalidate cached dependencies; public
`path` has no setter, and private-state mutation is outside the supported API.

At this initial checkpoint, workflow remained version 1 while `ml.spec` was explicit version 2;
shared requirement owners moved with live imports updated, without conversion or
compatibility modules. No inference mechanism, model loader or source-code converter
was introduced.

## Authorized cleanup: single workflow owner

The user requested moving the spec out of ML, deleting stale code and consolidating
segmentation postprocessing into tasks. This supersedes the earlier decision to
keep the version-1 prediction path alongside the new core.

- Replace `workflow/spec` with the approved call/reference implementation; move
  processors to `workflow/processing.py`, with no compatibility modules.
- Delete the old recipes, inference bindings, sampling/prediction runner and
  prediction flow. Retain native acquisition, I/O and the ingest flow.
- Import Prefect flows explicitly from `workflow.flows`; importing workflow specs
  and processors must remain independent of Prefect and Lightning.
- Move the current tensor interpretation functions from the user's renamed
  `ml/postprocess` directory into `ml/tasks/semantic_segmentation.py`, preserving
  their implementation and updating callback consumers.
- Put YAML behavior in its native loader/dumper methods rather than empty classes
  and separate callback functions. `!ref` stays an inert reference, not execution.
- Migrate current tests/examples/docs and preserve acquisition, I/O, laziness,
  serialization and task/callback coverage. Inference/model loading remain deferred.

Cleanup verification: 122 workflow core/acquisition/runtime tests, 13 persistence
and local Prefect ingestion tests, and 49 ML task/callback/native tiling tests passed
(184 affected tests). The restricted sandbox stalled at Zarr's async writer; the
same persistence tests and localhost integration passed outside that sandbox.
Ruff, whitespace checks, live import search and documented Python/YAML smoke passed.
Review caught the example's missing unpack operation; a packed-value test failed
with 6000 instead of 0.6, then passed after restoring explicit `.gs.unpack()`.
The final example also preserves lazy data and the caller's packed source values.

## Lean processor and executable examples

The subsequent cleanup removes the thin preprocessing/postprocessing subclasses.
One `Processor(spec, stage=...)` loads either supported stage through one operation
sequence, without duplicate callable lookup or compatibility aliases. Imported
callables remain opaque; only `Ref` targets resolve against runtime bindings.

The reader walkthrough now focuses on one input raster and one NDVI output, with
postprocessing shown independently in the same model document. The exhaustive grammar
checks live in tests, including a values fixture and accumulator lifecycle checks.
Inference execution and model loading remain deferred.

The readability sweep separates grid and metadata checks, validates source settings
before opening catalogues, shares local Zarr destination checks, and uses ExitStack
to close owned files. Short chunk comments describe each operation's purpose.
Validation: 129 core tests and 13 persistence/Prefect tests passed; workflow Ruff
and whitespace checks passed.

The model document is the only public YAML example. Separate ingestion and merge
recipes were removed: job parameters and runtime settings remain Python inputs.
Source requirements can configure acquisition without embedding catalogue or job
settings. Accumulator behavior remains covered by processor tests.
