# Workflow CLI and Execution Boundaries

Status: draft for user review.

## Goal

Expose GeoSave's complete workflow jobs through a small Typer command group while
making Prefect ownership visible in the Python package:

- flows are independently runnable jobs with serializable inputs and durable
  outputs;
- tasks are in-process steps composed by those jobs; and
- `model_spec.yaml` remains the only public workflow YAML document.

The initial CLI executes flows locally, waits for completion, and prints the
completed output path. Prefect deployments, remote submission, prediction, and a
generic flow launcher are outside this change.

## Package boundaries

The workflow package is organized around the distinction between runnable jobs
and their internal work:

```text
workflow/
├── specs/                    # model-owned declarations
├── configs/                  # serializable per-run values
├── tasks/
│   ├── process.py            # preprocessing and postprocessing tasks
│   ├── dense.py              # one label-aligned sample
│   ├── load.py
│   ├── save.py
│   └── catalog.py
└── flows/
    ├── ingest.py
    └── prepare_dense_data.py
```

`geosave_engine.workflow.flows` exports the independently runnable `ingest` and
`prepare_dense_data` flows. `geosave_engine.workflow.tasks` exports
`preprocess` and `postprocess` as reusable Prefect tasks alongside the existing
shared raster tasks. The workflow package root no longer exports a `dense`
module.

There are no compatibility aliases for the removed interfaces. The public
migration is:

```text
workflow.dense.prepare       -> workflow.flows.prepare_dense_data
workflow.flows.preprocess    -> workflow.tasks.preprocess
workflow.flows.postprocess   -> workflow.tasks.postprocess
```

`dense.validate_sample`, `run_stage`, and `invoke_call` are removed from the
public API.

## Model processing tasks

Preprocessing and postprocessing are not independently deployable jobs. Each is
one Prefect task with caching and result persistence disabled:

```python
preprocess(inputs: Mapping[str, Any], spec: ModelSpec) -> dict[str, Any]
postprocess(inputs: Mapping[str, Any], spec: ModelSpec) -> dict[str, Any]
```

Both tasks revalidate the model spec and execute their `StageSpec` declarations
sequentially through `CallSpec.invoke`. A private ordinary helper owns the
shared stage loop. It validates all references before invoking the first call,
copies the supplied state, binds each declared result for later references, and
returns only declared outputs.

`preprocess` retains its source-specific invariant: model source requirements
are applied only to external inputs consumed by the preprocessing stage.
`postprocess` supplies its inputs directly to the same stage loop. Individual
model calls no longer create separate Prefect task runs or overlap with one
another. Model-owned operation order, references, and literals remain in
`model_spec.yaml`.

The existing processing flow modules, public `run_stage`, and the generic
`invoke_call` Prefect task are deleted.

## Runnable flows

### Ingest

`ingest` loads all required model sources on one explicit anchor and writes one
completed Zarr stack:

```python
ingest(
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str,
    sources: dict[str, dict[str, JsonValue]] | None = None,
) -> str
```

When `sources` is omitted, the flow creates a default `SourceConfig` for every
source declared by the model spec. When supplied, source names must exactly
match the model requirements. Ingest has no concurrency parameter because it
produces only one stack.

### Prepare dense data

`prepare_dense_data` discovers label rasters, prepares one label-aligned sample
per raster, and publishes the completed samples as a GeoParquet manifest:

```python
prepare_dense_data(
    labels: str,
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
    sources: dict[str, dict[str, JsonValue]] | None = None,
    max_concurrency: PositiveInt = 1,
) -> str
```

The flow module owns primitive validation, stable label discovery, bounded task
submission, and final manifest publication. `tasks/dense.py` owns the private
per-label Prefect task and sample validation. The task keeps one concurrency
slot from STAC search through lazy raster computation and completed Zarr
persistence.

`max_concurrency` is a direct per-run limit on simultaneous complete sample
ingestions. It defaults to `1`, so separate samples cannot overlap unless the
caller explicitly opts into parallelism. The flow initially submits at most the
configured number of samples. Each completion frees one slot and permits one
new submission. This keeps the queue utilized without exceeding the limit.

Concurrency is not stored in `SourceConfig`, and GeoSave does not require or
create a named Prefect global concurrency limit. The existing
`SourceConfig.concurrency` field and `source_concurrency` context manager are
removed. The limit does not claim to count individual HTTP or GDAL requests
within one sample; it controls the complete concurrent STAC ingestion units
owned by this flow.

## Persistence and failure behavior

All primitive configuration, model requirements, source bindings, label
discovery, and `max_concurrency` are validated before pixel work begins.

Each sample uses the existing atomic local Zarr writer. It computes into a
temporary sibling store and renames the store into place only after every chunk
succeeds. A failed computation cleans its temporary store. An existing sample
is validated against the current label and model source requirements before it
is reused.

After the first sample failure, the flow submits no new labels. Tasks already in
flight may finish because interrupting an active raster write is unsafe. Their
atomic completed outputs remain reusable on the next run. The existing manifest
is replaced only after every requested sample succeeds, so it never advertises
a partial run.

CLI input or Pydantic validation errors occur before raster work and are shown
as command usage errors. Runtime flow failures produce a nonzero exit and retain
their Prefect error context. A command prints its output path only after the
flow completes successfully.

## CLI contract

The root Typer application mounts a `workflow` command group with two concrete
commands:

```text
geosave workflow ingest
geosave workflow prepare-dense-data
```

There is no generic `workflow run`, arbitrary `--param` grammar, deployment
selector, or job-parameter YAML file.

### Ingest command

```bash
geosave workflow ingest \
  --anchor '{"kind":"raster","path":"data/reference.tif"}' \
  --output data/raw.zarr \
  --spec configs/model_spec.yaml
```

The command exposes:

- required `--anchor` JSON validated as the existing discriminated
  `AnchorConfig`;
- required `--output` local Zarr path;
- required `--spec` model-spec path; and
- optional `--sources` JSON validated as a mapping of source names to existing
  `SourceConfig` values.

The guide includes raster, GeoJSON, and coordinate anchor examples.

### Prepare-dense-data command

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec configs/model_spec.yaml
```

The command exposes:

- required `--labels`, `--output`, and `--spec` paths;
- `--pattern`, defaulting to `**/*.tif`;
- `--max-concurrency`, defaulting to `1`; and
- optional `--sources` JSON using the same schema as ingest.

Structured values use JSON because it is already the primitive serialization
shape accepted by the flow and Pydantic models. GeoSave does not introduce a
custom dotted-key or `key=value` language. Omitting `--sources` selects default
settings for every source in the model spec. An advanced invocation may supply
the existing schema directly:

```bash
--sources '{
  "sentinel_2_l2a": {
    "query": {"max_items": 8},
    "load": {"chunks": {"x": 512, "y": 512}}
  }
}'
```

The CLI uses a small shared JSON parser around Pydantic `TypeAdapter`; it does
not duplicate the model fields or their validation.

## User guide

Add `docs/guides/workflows.md` to the Zensical navigation. It leads with each
command's purpose, input, and output, then provides:

- minimal runnable commands;
- every Typer option and its default;
- all supported anchor shapes;
- default and overridden source settings;
- the serial safety default and explicit parallel opt-in;
- atomic-write, failure, and resume behavior; and
- equivalent Python calls for users composing their own flows.

The generated Typer help remains the concise command reference:

```bash
geosave workflow --help
geosave workflow ingest --help
geosave workflow prepare-dense-data --help
```

The guide explains what the workflows do and what they produce. Internal queue
mechanics and exhaustive edge cases remain in tests.

## Tests

Tests continue to mirror source ownership:

- `tests/workflow/tasks/test_process.py` covers sequential preprocessing and
  postprocessing, input validation, source selection, rebinding, `None`, and
  lazy values;
- `tests/workflow/tasks/test_dense.py` covers one sample, atomic persistence,
  validation, and reuse;
- `tests/workflow/flows/test_prepare_dense_data.py` covers stable discovery,
  exact source bindings, default serial execution, the explicit concurrency
  ceiling, stopping submissions after failure, resumability, and manifest
  replacement;
- `tests/workflow/flows/test_ingest.py` covers default and explicit source
  settings plus the existing anchor variants;
- `tests/workflow/configs/test_source.py` proves the removed concurrency field
  is rejected; and
- `tests/cli/commands/test_workflow.py` covers command help, defaults, JSON
  validation, flow arguments, successful output, and nonzero failure behavior.

Documentation examples are checked against the actual command names and option
defaults. Focused workflow and CLI tests run before the broader affected suite,
followed by scoped Ruff, BasedPyright, a Zensical build, and
`git diff --check`.

## Breaking changes

- `geosave_engine.workflow.dense` is removed.
- `dense.prepare` is replaced by `flows.prepare_dense_data`.
- `dense.validate_sample` becomes private task support.
- `flows.preprocess` and `flows.postprocess` move to `tasks` and change from
  Prefect flows to Prefect tasks.
- `run_stage` and `invoke_call` are removed.
- `SourceConfig.concurrency` and named Prefect global-limit behavior are
  removed.
- Individual model calls no longer appear as separate Prefect task runs or run
  concurrently.

No aliases or parallel execution paths preserve these obsolete interfaces.

## Deliberately deferred

- Prefect deployment creation and remote flow submission.
- Cross-run or deployment-wide concurrency limits.
- Exact HTTP requests-per-second throttling within one sample ingestion.
- Prediction and training commands.
- Remote Zarr destinations.
- Job-parameter YAML or JSON files.
- A generic flow or task registry.
