# Workflow

Workflow reads a model spec to prepare compatible data, run its processing steps
and save results. The YAML belongs to the model; job arguments remain Python inputs.

| Responsibility | Belongs to |
| --- | --- |
| Required bands, dimensions, STAC collection/endpoints and model-specific processing | `model_spec.yaml` |
| Requested area, dates, input files and output destination | Job parameters |
| STAC query/load options, chunking, device, batching and Prefect settings | Runtime configuration |

A source name such as `optical` identifies the model's required data. Runtime
configuration binds that name to query/load options; the job chooses the region,
time range or existing input files. The spec remains the same across those jobs.

| API | Input | Output |
| --- | --- | --- |
| `Processor` | Model spec, stage name and named Python objects | Inputs plus named step results |
| `acquire` | STAC sources and a region | Raster stack (`xarray.DataTree`) |
| `open_rasters` | Raster objects or file paths | Open raster stack |
| `write_stack` | Raster stack and destination | Completed local Zarr path |
| `ingest` | Model spec, source settings, region settings and destination | Completed Zarr path, monitored by Prefect |

## Run processing steps

```python
from geosave_engine.workflow import Processor

prepare = Processor.load("model_spec.yaml", stage="preprocessing")
features = prepare({"optical": optical})["features"]
```

The spec names the inputs and calls needed by the model. Preprocessing and
postprocessing run independently and return native Python objects. Importing a
processor does not import Prefect or Lightning. Start with the
[segmentation workspace example](../templates/tasks/semantic_segmentation/supervised/README.md);
see the [spec reference](spec/README.md)
for the declaration rules. Model loading and inference execution remain deferred.

## Acquire and save rasters

```python
from geosave_engine.workflow import acquire
from geosave_engine.workflow.io import open_rasters, write_stack
from geosave_engine.workflow.spec import ModelSpec

spec = ModelSpec.load("model_spec.yaml")
raw = acquire(sources, anchor, requirements=spec.sources)
path = write_stack(raw, "raw.zarr")

with open_rasters(path) as opened:
    prepared = prepare(opened.gs.rasters)
    # Consume lazy results while their source files remain open.
    features = prepared["features"].compute()
```

`acquire` uses configured native STAC sources and a `GeoAnchor`. Source requirements
select bands and validate the acquired rasters. `write_stack` returns after the
new Zarr store is complete; it rejects existing destinations.

## Monitor acquisition with Prefect

```python
from geosave_engine.workflow.flows import ingest

source_config = {
    "optical": {
        "query": {"max_items": 20},
        "load": {"groupby": "time", "chunks": {"x": 1024, "y": 1024}},
    }
}
path = ingest(
    sources=source_config,
    anchor=anchor_parameters,
    output="raw.zarr",
    spec="model_spec.yaml",
)
```

The required model spec declares each source's collection and ordered endpoints:

```yaml
sources:
  optical:
    variables: [nir, red]
    collection: sentinel-2-l2a
    endpoints:
      - https://earth-search.aws.element84.com/v1/
      - https://planetarycomputer.microsoft.com/api/stac/v1/
```

Runtime source names must exactly match the model sources; only `query` and `load`
are accepted, and both default to `{}`. The flow validates every binding before
opening a catalogue, then probes each endpoint's root and required collection in
order. Only transport failures, HTTP 404/5xx and missing collections try the next
endpoint. Authentication errors and malformed documents stop immediately. Empty
searches and later asset-read failures do not trigger endpoint failover. Asset
authentication or URL signing remains the caller's responsibility; direct Python
`acquire` still accepts caller-configured `StacSource` objects.

`anchor_parameters` and `output` describe the requested job. The flow builds native
objects inside the worker, acquires the sources, and saves the result. No separate
flow YAML is needed. Requirements for existing rasters can omit both collection
and endpoints; those sources cannot be acquired through `ingest`.

Prefect owns monitoring and task execution. It does not define the model contract;
the same processing stages remain usable from ordinary Python.

ML task behavior, including segmentation tensor interpretation, lives in `ml.tasks`.
Workflow owns configuration and execution of the declared processing steps.
