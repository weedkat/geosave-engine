# Workflow Layering Design

## Context

GeoSave currently organizes workflow code into `workflow.ingestion` and
`workflow.training_data`. That layout replaced the earlier technical
`configs`, `flows`, and `tasks` packages, but it does not scale cleanly:
every new runnable job needs a new domain package, `training_data` claims a
broader lifecycle than it implements, and callers must know which domain
package owns each Prefect entry point.

The workflow package should instead have stable technical layers. New jobs
extend those layers without changing the package taxonomy. Model declarations
remain separate because they are consumed by workflow, release, templates, and
eventually serving.

## Goals

- Restore `workflow.configs`, `workflow.flows`, and `workflow.tasks` as the
  stable package structure.
- Keep Prefect flows thin and independently runnable.
- Keep reusable work, persistence, and manifest behavior out of flow modules.
- Preserve `ModelSpec` ownership of raster acquisition and preprocessing.
- Expose supported interfaces explicitly from each technical package.
- Preserve all current ingestion, dense preparation, metadata, resume,
  concurrency, and persistence behavior.
- Make future workflows additive: a new job adds a flow module, the task
  modules it needs, and runtime config models only when primitive flow
  parameters are insufficient.

## Non-goals

- Do not restore removed prediction, postprocessing, or source-configuration
  APIs.
- Do not move model recipes back into workflow config.
- Do not implement the deferred training Dataset/DataModule.
- Do not change whether dense preparation stores acquired rasters or processed
  stage outputs; that data-contract decision needs its own design and tests.
- Do not add compatibility aliases for the Alpha domain-package imports.
- Do not decorate every function as a Prefect task merely because it lives in
  `workflow.tasks`.

## Chosen structure

```text
geosave_engine/workflow/
├── __init__.py
├── configs/
│   ├── __init__.py
│   ├── base.py
│   └── anchor.py
├── flows/
│   ├── __init__.py
│   ├── ingest.py
│   └── prepare_dense_data.py
└── tasks/
    ├── __init__.py
    ├── dense.py
    ├── manifest.py
    ├── sample.py
    └── stack.py
```

The technical layers are stable; filenames inside them describe concrete
jobs or work units. There is no `workflow.training_data` umbrella and no
separate package per current flow.

The root `workflow` package does not duplicate layered exports. Supported use
is explicit:

```python
from geosave_engine.workflow.configs import AnchorConfig
from geosave_engine.workflow.flows import ingest, prepare_dense_data
from geosave_engine.workflow.tasks import prepare_dense_sample
```

## Layer responsibilities

### Configs

`workflow.configs` owns strict Pydantic models that turn deployment-safe
primitive values into native runtime objects. `ConfigModel` centralizes the
shared frozen, strict validation policy. `AnchorConfig` and its concrete
variants retain their `open()` behavior because constructing a native anchor
is the capability represented by those values.

Configs do not load a `ModelSpec`, call tasks, or choose orchestration policy.
STAC queries and raster requirements remain in `geosave_engine.model_spec`;
they are model declarations, not flow invocation config.

### Flows

`workflow.flows` exports only independently runnable Prefect flows:

- `ingest` validates an anchor invocation, loads the model spec, acquires the
  declared rasters, and publishes one stack.
- `prepare_dense_data` validates the complete invocation before submission,
  discovers labels, applies the bounded-concurrency policy, collects completed
  sample paths in deterministic order, and publishes the manifest.

Flows own sequencing, fail-fast validation order, concurrency, and propagation
of task failures. They do not implement storage layouts, metadata table
parsing, sample validation, or atomic publication.

One flow may invoke a task normally when ordering matters and submit it only
when concurrent execution is intentional. No flow is introduced merely as a
wrapper around another flow.

### Tasks

`workflow.tasks` owns reusable workflow work units and their cohesive ordinary
Python operations:

- `stack.py` owns atomic publication of one acquired raster stack.
- `dense.py` owns the Prefect `prepare_dense_sample` task and validation of a
  resumable dense sample.
- `sample.py` owns the GeoTIFF/Zarr dense-sample storage contract, including
  open, write, format validation, and atomic publication.
- `manifest.py` owns label discovery, sample-path derivation, metadata-table
  matching, manifest schema, and atomic GeoParquet replacement.

The package name `tasks` describes workflow work, not a requirement that every
function carry `@task`. A function receives a Prefect task decorator only when
the flow benefits from task identity, retries, caching policy, or submission.
Small helpers stay ordinary Python beside the work unit whose invariant they
support.

Public reusable operations use normal names and concise Google-style
docstrings. Narrow wiring helpers may remain private. `tasks.__init__` exports
only supported composition points rather than every helper in every task
module.

## Dependency direction

```text
CLI and generated scripts
    -> workflow.flows
        -> workflow.configs
        -> workflow.tasks
            -> model_spec
            -> geodata

release -> model_spec + ml
```

`workflow.tasks` never imports `workflow.flows`. Configs do not import either
flows or tasks. Model-spec and geodata packages do not depend on workflow.
This prevents orchestration concerns from leaking into reusable model and data
behavior.

## Data flow

### Explicit-anchor ingestion

1. `flows.ingest` loads and validates the model specification.
2. It checks that every required raster has an acquisition recipe before
   opening an anchor that may itself require I/O.
3. `AnchorConfig` parses the primitive mapping and opens a native `GeoAnchor`.
4. `ModelSpec.load_rasters(anchor)` performs model-owned acquisition and
   raster validation.
5. `tasks.stack.write_stack` atomically publishes the completed Zarr stack.

### Dense data preparation

1. `flows.prepare_dense_data` validates concurrency, model recipes, the
   reserved label name, discovered labels, and optional metadata before
   submitting any work.
2. It submits `tasks.prepare_dense_sample` up to the configured concurrency
   bound.
3. Each sample task opens its label, derives the anchor, calls
   `ModelSpec.load_rasters(anchor)`, validates resumable output, and delegates
   publication to `tasks.sample`.
4. The flow stops submitting new samples after a task failure.
5. Once every sample succeeds, `tasks.manifest.write_manifest` publishes the
   ordered GeoParquet manifest synchronously.

This restructuring does not add `ModelSpec.preprocess()` to dense preparation.
Acquisition and preprocessing remain separate model capabilities until the
training sample contract decides whether persisted samples contain raw
required rasters, preprocessing outputs, or both.

## Prefect policy

`prepare_dense_sample` remains a Prefect task with no cache and no persisted
Prefect result because it is the concurrency-bound unit. Stack and manifest
publication remain synchronous operations unless concrete retry or execution
visibility requirements justify decorating them later. Prefect futures do not
cross into storage or model-spec modules.

Future few-shot, autoregressive, or prediction programs follow the same rule:
their iterative behavior belongs in reusable task/program modules, while a
Prefect flow exists only for an independently runnable, durable job. Pure
tensor model calls remain in `ml` rather than becoming workflow policy.

## Public API and migration

The supported workflow interfaces are:

```python
# Deployment-safe invocation models
from geosave_engine.workflow.configs import (
    AnchorConfig,
    CoordinateAnchorConfig,
    GeoJSONAnchorConfig,
    RasterAnchorConfig,
)

# Deployable jobs
from geosave_engine.workflow.flows import ingest, prepare_dense_data

# Reusable submitted work
from geosave_engine.workflow.tasks import prepare_dense_sample
```

CLI commands, generated scripts, documentation, and tests move to these paths.
The following Alpha imports are removed without aliases:

```text
geosave_engine.workflow.ingestion
geosave_engine.workflow.training_data
```

Artifact functions remain available from their owning task modules for direct
testing and deliberate advanced use, but are not all re-exported from
`workflow.tasks`:

```python
from geosave_engine.workflow.tasks.manifest import write_manifest
from geosave_engine.workflow.tasks.sample import open_sample, write_sample
from geosave_engine.workflow.tasks.stack import write_stack
```

## Physical migration

```text
workflow/ingestion/anchor.py
    -> workflow/configs/base.py + workflow/configs/anchor.py
workflow/ingestion/flow.py
    -> workflow/flows/ingest.py + workflow/tasks/stack.py
workflow/training_data/dense.py
    -> workflow/flows/prepare_dense_data.py + workflow/tasks/dense.py
workflow/training_data/sample.py
    -> workflow/tasks/sample.py
workflow/training_data/manifest.py
    -> workflow/tasks/manifest.py
```

Tests mirror the resulting source tree. CLI and template tests assert the new
layered imports so a generated workspace cannot retain a deleted path.

## Error handling and invariants

The move preserves current exception types and validation order. In
particular:

- invalid metadata fails before the first sample submission;
- missing recipes fail before remote anchor or STAC access;
- existing complete samples are validated and reused;
- incomplete or incompatible samples fail rather than being overwritten;
- sample and stack writes publish atomically and preserve existing targets;
- the manifest is replaced only after all samples succeed;
- active sample preparation never exceeds `max_concurrency`;
- raster pixels remain lazy until persistence requires computation.

## Testing

Tests move to `tests/workflow/configs`, `tests/workflow/flows`, and
`tests/workflow/tasks`, mirroring source ownership. The migration keeps the
existing behavioral coverage and adds import-surface assertions:

- config tests cover primitive parsing and native anchor opening;
- flow tests cover validation order, bounded submission, failure stopping,
  deterministic output, and CLI forwarding;
- task tests cover sample preparation, resume validation, storage round trips,
  metadata matching, manifest publication, and atomic failure behavior;
- stale-import checks cover source, tests, templates, and active docs;
- a clean-export import smoke test proves the layered packages do not rely on
  unrelated working-tree files.

Final verification runs the focused workflow/CLI/template tests, Ruff,
BasedPyright on the changed packages, the complete non-slow test suite, and a
package build.

## Alternatives rejected

### Domain packages per workflow

Keeping `workflow.ingestion` and `workflow.training_data` concentrates current
files but makes the taxonomy change whenever a new workflow appears. It also
claims domains before their full contracts exist. This is the structure being
replaced.

### Flat workflow package

Putting flow, config, task, storage, and manifest modules directly under
`workflow` makes ownership ambiguous as the number of jobs grows. Root modules
would encode implementation details rather than a stable public structure.

### Decorate every operation as a Prefect task

This would couple ordinary storage and validation code to Prefect, enlarge the
runtime state surface, and encourage submitting work that does not need
concurrency. Decorators remain an execution choice, not the package taxonomy.
