# Model workflows

```python
from geosave_engine.workflow import ingest, predict

raw = ingest(sources, anchor, output="raw.zarr")
outputs = predict(raw, model="artifacts/model", output="predictions")
```

Two Prefect flows accept primitive settings and paths. Workers open native STAC
clients, rasters and saved models. Ordinary Python functions own acquisition,
preprocessing, inference and postprocessing, so preparation remains reusable
outside a prediction run.

## Model YAML

- `ModelSpec` contains `schema_version: 1`, `sources`, `preprocessing`,
  `inference` and optional `postprocessing`. Save/load uses `model_spec.yaml`,
  safe YAML with duplicate-key rejection, and ordinary Pydantic validation.
  A YAML file or model artifact directory can be supplied. Weights are preserved.
- `sources` declares required variables, dimensions, dtypes, grid and metadata.
  `attrs.root`, `attrs.data_vars` and `attrs.coords` use native attrs namespaces.
  Each namespace has sibling `models` and `foreign` requirements. These validate
  existing metadata; coordinate values are ordinary xarray coordinates.
- Each preprocessing recipe has a starting `raster`, optional ordered `variables`
  and ordered `operations`. Operations select exactly one native `method`, such
  as `gs.unpack` or `assign_coords`, or an installed Python `call`. Primitive
  constants belong in `kwargs`; other raster dependencies belong in `inputs`.
  Native methods and custom calls share one execution path. No operation registry
  or expression language is required.
- Recipes form an acyclic graph and produce Datasets. Preprocessing retains
  selected sources and adds named recipe outputs in one native DataTree. Branches
  isolate data and metadata while Dask pixels remain lazy. Spatial coordinate
  names on a shared root are reserved against DataTree group collisions.
- `inference.inputs` binds model argument names to rasters, ordered variables,
  CHW/TCHW layout, dtype and optional normalization. `inference.tiling` reuses
  native `Tiles` and `TileMerger`; optional `time_window` reuses `window_stack`.
  Unbound reference dates and static scalar time coordinates do not constrain
  temporal sampling. Inputs must share the reference spatial grid.
- Inference accepts dense `[B, C, H, W]` model output. Tiles are merged as logits
  before segmentation or custom postprocessing. Each temporal window produces
  its own Dataset with coverage metadata. Model training state is restored after
  inference, including errors. Constant context and a per-tile context callable
  support additional modalities without introducing an adaptation algorithm.

## Runtime inputs

- `ingest(sources, anchor, *, output, spec=None)` returns a completed raw Zarr
  path. Source names map to STAC `url`, `collection`, optional native `query`
  and `load` settings. An optional model spec narrows acquisition requirements;
  ingestion otherwise stays independent of a model.
- `anchor.type: coordinates` uses named latitude/longitude, shape and resolution.
  `anchor.type: geojson` uses a file path and either shape or resolution. Both
  accept CRS and timespan. Native GeoAnchor constructors own grid behavior;
  a geometry anchor covers the geometry's extent and does not mask pixels.
- STAC `load.with_properties` delegates aliases, dtype, nodata, units and grouping
  to odc-stac. Loaded auxiliary properties become coordinates along `time`, such
  as `sun_azimuth`. `item_properties` retains separate acquisition provenance.
- `predict(inputs, *, model, output, ...)` accepts a saved raw stack path or
  raster names mapped to paths. The default model loader is native
  `ModelChain.from_pretrained`; an import path can select another native loader.
  Model YAML defaults to the model artifact directory. Model options, constant
  context, device and batch size are primitive settings; a per-tile context
  function is supplied by import path.
- Optional `prepared_output` completes a local Zarr checkpoint and reopens it for
  inference. Prediction returns completed per-window Zarr paths. Raw, prepared
  and prediction outputs are staged locally before publishing their paths;
  existing destinations are rejected. Files open with Dask chunks and owned
  handles close after prediction. Caller-owned native objects remain supported
  by ordinary domain functions.

## Orchestration and scope

Plain Prefect flows coordinate bounded acquisition tasks and preparation,
inference, postprocessing and persistence tasks. Live models and lazy graphs stay
inside a worker process; automatic cache/result persistence is disabled for them.
Native Prefect task-runner configuration remains available. Completed files form
the durable boundary between runs.

The initial implementation supports local output paths and dense raster models.
Separate GPU deployments, few-shot weight adaptation, non-dense outputs and
remote publishing remain future work. Merged outputs occupy memory per run.

## Verification

Exercise safe YAML round trips, native methods, ordered band selection, attrs
requirements, lazy branches, real local STAC property coordinates, coordinate
and GeoJSON anchors, saved-model loading, temporal multimodal context, overlap
reconstruction, model-state restoration, failure propagation and persisted
outputs. Run the offline example and the default test suite. Preserve unrelated
working-tree changes and leave implementation uncommitted for review.
