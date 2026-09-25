# Workflow package redesign

## Intent

Rebuild `geosave_engine.workflow` around four visible ownership areas:

```text
workflow/
  specs/    model_spec.yaml declarations and persistence
  configs/  Pydantic parsing for primitive flow parameters
  tasks/    small Prefect tasks with one observable operation
  flows/    deployable Prefect orchestration
```

The package must make each workflow concept easy to locate. Behavior belongs to
one model, class, task, or flow rather than scattered module-level helper
functions. Obsolete modules are deleted instead of retained through aliases or
adapters.

The first implementation supports model requirements, preprocessing, validated
inference calls, and the ingestion flow. Postprocessing, sampling, tiling,
merging, model loading, and inference execution remain deliberately unresolved.

## Ownership and package structure

```text
src/geosave_engine/workflow/
  __init__.py
  specs/
    __init__.py
    base.py
    model.py
    sources.py
    preprocessing.py
    postprocessing.py
  configs/
    __init__.py
    base.py
    anchor.py
    source.py
    ingest.py
  tasks/
    __init__.py
    load.py
    preprocess.py
    save.py
  flows/
    __init__.py
    ingest.py
```

`specs` contains only portable declarations stored in `model_spec.yaml`.
`configs` contains runtime values supplied to flow deployments. `tasks` contains
actual Prefect tasks. `flows` is the only orchestration layer and contains the
deployable flow entry points.

The package root does not flatten these interfaces into broad convenience
imports. Callers import the concept from its owning package, such as
`workflow.specs.ModelSpec`, `workflow.configs.IngestConfig`, or
`workflow.flows.ingest`.

The following obsolete files are deleted:

- `workflow/spec/`
- `workflow/ingestion.py`
- `workflow/io.py`
- `workflow/processing.py`
- `workflow/runtime.py`
- `workflow/flows.py`
- `workflow/README.md`

The nested `workflow/spec/README.md` is also removed. Durable design documents
live under `docs/`, outside the import package.

## Model specification

`ModelSpec` remains the owner of `model_spec.yaml`. It provides strict YAML
load/save behavior, duplicate-key rejection, local artifact path resolution,
and Pydantic validation. Loading the document never imports or invokes declared
callables.

The document has these top-level fields:

```yaml
schema_version: 2
sources: {}
preprocessing: {}
inference: {}
postprocessing: {}
```

- `sources` maps external names to `RasterRequirement` declarations.
- `preprocessing` is an ordered mapping of names to `OperationSpec`.
- `inference` is an ordered mapping of model argument names to the same
  `OperationSpec` interface.
- `postprocessing` is an empty `PostprocessingSpec`. Non-empty declarations are
  rejected until its semantics are designed.

The existing output-legend execution and `OutputSpec` are removed with the old
postprocessing implementation.

### Calls and references

One operation names one complete return value:

```yaml
preprocessing:
  reflectance:
    call: !ref raw.gs.unpack

  normalized:
    call: project.preprocessing.normalize
    kwargs:
      raster: !ref reflectance
      mean: [0.1, 0.1, 0.1, 0.1]
      std: [0.2, 0.2, 0.2, 0.2]

inference:
  image:
    call: !ref normalized.gs.to_tensor
    kwargs:
      dtype: float32
```

`call` is either an importable module-level callable or an inert `Ref` to a
supplied callable or bound method. `kwargs` contains finite YAML primitives and
explicit references. Plain strings always remain literals. There is no
registry, expression language, implicit receiver, or inferred string reference.

An operation resolves its call and arguments against the current named values,
invokes it once, and binds the complete return value to the enclosing name.
Rebinding resolves the previous value first. A call returning `None` binds
`None`. Parsing never imports or executes calls.

Reference validation, traversal, and resolution are owned by `Ref` and
`OperationSpec`; they are not exposed as unrelated module-level helper
functions. YAML path handling is owned by `ModelSpec`.

### Preprocessing and inference meaning

Preprocessing executes over native xarray values and should remain lazy. It owns
semantic pixel preparation such as variable selection, ordering, unpacking,
normalization, masking, and lazy xarray dtype conversion.

Inference declarations describe per-sample model arguments. They reuse the call
pattern instead of introducing fields for variables, layout, dtype, or
normalization. A future sampler will replace referenced prepared rasters with
bounded samples, then execute inference calls. This is where an explicit
`to_tensor` call belongs.

The initial implementation validates and round-trips inference calls but does
not execute them. It contains no `infer` task, prediction flow, tensor batching,
temporal windowing, tiling, merging, model loading, device selection, or
autocast policy.

## Tensor conversion

`to_tensor` is a representation conversion, not an implicit preprocessing
policy. Dataset, DataArray, and DataTree accessors accept a YAML-friendly dtype
name as well as a native `torch.dtype`:

```python
tensor = sample.gs.to_tensor(dtype="float32")
```

Its responsibilities are limited to:

1. arranging native dimensions in the established deterministic order;
2. materializing the bounded native value as contiguous tensor storage; and
3. casting to an explicitly requested tensor dtype.

When `dtype` is omitted, `to_tensor` preserves the prepared xarray dtype rather
than silently casting to float32. Unsupported NumPy-to-Torch dtype conversion
raises a clear error. Semantic dtype changes may be declared lazily during
preprocessing with xarray `astype`. GPU execution precision, mixed precision,
and autocast remain future runtime concerns.

`TileDataset(dtype=...)` is removed because it currently introduces an implicit
dtype policy at sampling time. This removal does not otherwise redesign
`Tiles`, `TileDataset`, or `TileMerger`; their replacement or evolution is a
separate design task.

Preprocessing produces sample-ready lazy xarray values, not tensors. Tensor
conversion must occur only after temporal/spatial sampling has bounded the
pixels being read.

## Flow configuration

Flow entry points retain primitive, deployment-friendly parameters. Each flow
builds one Pydantic config at its entrance:

```python
config = IngestConfig.model_validate(
    {
        "sources": sources,
        "anchor": anchor,
        "output": output,
        "spec": spec,
    }
)
```

Unknown fields, non-finite values, invalid paths, and invalid combinations are
rejected before task submission.

`AnchorConfig` is a discriminated union selected by `kind`:

```yaml
kind: coordinates
latitude: -6.2
longitude: 106.8
shape: [512, 512]
resolution: 10
```

```yaml
kind: geojson
path: region.geojson
resolution: 10
```

Coordinate anchors require latitude, longitude, shape, and resolution. GeoJSON
anchors require a local `.json` or `.geojson` path and exactly one of shape or
resolution. Both support CRS and timespan; GeoJSON supports padding. Pydantic
normalizes YAML lists to native tuples. The selected config owns conversion to
`GeoAnchor` through a method on the config model.

`SourceConfig` parses per-run STAC query and load settings. Model-owned
collection identity, endpoint priority, and raster requirements remain in
`RasterRequirement`. `IngestConfig` checks local `.zarr` output syntax and owns
the complete primitive parameter set. Cross-checking source names against the
loaded `ModelSpec` happens in the flow before any task is submitted.

## Tasks and flows

The initial ingestion workflow uses Prefect's monoflow pattern: one deployable
flow coordinates a small task graph. A future prediction flow may use a
separate GPU deployment when its sampling and inference interfaces are known.

`tasks/load.py` exposes one `@task` named `load_raster`. It receives a source
name, `SourceConfig`, `AnchorConfig`, and `RasterRequirement`. The function is a
short call into a cohesive loader class in the same module. That class owns
STAC query construction, endpoint probing/failover, native source construction,
band selection, and requirement validation. Native anchors, clients, and
sources are constructed inside the task.

`tasks/save.py` exposes one `@task` named `save_stack`. It combines the named
rasters, writes a fully computed local Zarr store through native geodata I/O,
and publishes the destination only after the staged write succeeds. Existing,
remote, or non-Zarr destinations are rejected. A failed write leaves no partial
destination.

`tasks/preprocess.py` exposes one `@task` named `preprocess`. The decorated
function is short; a cohesive executor in the same module owns validation,
callable loading, reference resolution, and ordered preprocessing execution.
It returns fresh named bindings and preserves native object identity and Dask
laziness. It never executes inference or postprocessing declarations.

`flows/ingest.py` exposes the `ingest` flow. It validates `IngestConfig`, loads
`ModelSpec`, rejects missing or extra source bindings, submits one
`load_raster` task per source, and passes the completed named results to
`save_stack`. The flow returns the completed Zarr path.

Lazy xarray task results use `cache_policy=None` and `persist_result=False`.
Completed geospatial files, not clients, native anchors, loaded models, lazy
graphs, or accumulators, are durable workflow seams.

No public or private workflow symbol uses the word `acquire`.

## Errors

Pydantic validation errors identify malformed flow parameters and model fields
before remote work. Source loading tries the next endpoint only for transport
failures, HTTP 404/5xx responses, or a missing required collection.
Authentication errors, malformed catalogue documents, invalid configuration,
empty searches, and lazy asset failures propagate without fallback.

Tasks add only source or stage context when an underlying error lacks it. They
do not broadly translate native xarray, STAC, filesystem, or PyTorch errors.
Preprocessing validates all required references and active imports before its
first operation runs.

## Testing

Implementation follows test-driven development. Tests mirror the new package
structure and exercise public behavior rather than removed source text.

- `tests/workflow/specs/` covers strict YAML/Python round trips, duplicate keys,
  inert references, preprocessing and inference calls, empty-only
  postprocessing, source requirements, and invalid paths.
- `tests/workflow/configs/` covers `kind` dispatch, field normalization, invalid
  anchor combinations, primitive source settings, output paths, and config
  validation before I/O.
- `tests/workflow/tasks/` covers endpoint behavior with the real local STAC
  fixture, one-raster loading, requirement validation, lazy preprocessing,
  explicit call/rebinding behavior, and staged persistence.
- `tests/workflow/flows/` covers primitive flow parameters, one task per source,
  failure propagation, and a completed local Zarr artifact.
- Geodata model-I/O and tile-dataset tests cover dtype-preserving `to_tensor`,
  explicit string/native dtype casts, deterministic axis/variable order, and
  removal of sampling-time dtype conversion.

Focused package tests and Ruff run throughout implementation. Final verification
runs the complete default test suite, repository Ruff checks excluding notebooks,
format checks, `git diff --check`, and fresh-process import smoke tests proving
that specs/configs do not import Prefect and only task/flow packages do.

## Breaking changes

- `geosave_engine.workflow.spec` becomes `geosave_engine.workflow.specs`.
- Root workflow convenience exports are removed.
- `acquire`, `stac_config`, `open_anchor`, `open_sources`, `open_rasters`,
  `write_stack`, and `Processor` are removed.
- `workflow.flows` becomes a package containing `workflow.flows.ingest`.
- The current output legend/postprocessing execution is removed.
- `to_tensor()` no longer silently returns float32 and accepts string dtype
  names for explicit model input conversion.
- `TileDataset` no longer accepts `dtype`.

There are no compatibility aliases, migration adapters, duplicate execution
paths, workflow-local I/O wrappers, or YAML schema converters.

## Deferred design

The following require separate design work after this implementation:

- temporal and spatial sampling;
- `Tiles`, `TileDataset`, and `TileMerger` evolution;
- inference call execution and model loading;
- batching, device placement, and mixed precision;
- the prediction flow and GPU deployment;
- postprocessing, output legends, and reconstructed output persistence;
- orchestration between independently deployed ingestion and prediction flows.
