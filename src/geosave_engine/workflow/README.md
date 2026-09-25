# Model workflows

```python
from geosave_engine.workflow import predict

paths = predict(
    {"optical": "image.zarr", "terrain": "elevation.tif"},
    model="artifacts/model",
    output="predictions",
    batch_size=8,
)
```

Public Prefect flows accept primitive values and paths. They construct native
clients, anchors, raster objects and models inside the worker, and return paths
to completed results. `predict` loads `model_spec.yaml` beside the model unless
an explicit `spec` path is supplied. The default model loader is the existing
`ModelChain.from_pretrained`; `model_loader` and `model_options` select another
installed loading function and its primitive options when the artifact needs it.

Run a complete offline numerical example without a catalog, saved weights or a
Prefect server:

```bash
uv run python -m geosave_engine.workflow.examples.reflectance
```

## Readable model operations

```yaml
preprocessing:
  reflectance:
    raster: optical
    variables: [nir, red]
    operations:
      - method: gs.to_nan
      - method: gs.unpack
```

`variables` selects and orders a recipe's bands before its operations. `method`
invokes an existing Dataset or accessor method, including `isel`, `set_coords`
and `assign_coords`, with native arguments under `kwargs`. Each operation must
return a Dataset. Custom Python uses `call: my_model.prepare`; the current
Dataset is its first argument. `inputs` binds additional named rasters to
keyword arguments. There is no operation registry or expression language.

Recipes form a dependency graph and produce a native DataTree containing the
selected sources and recipe outputs. Branches cannot mutate caller data or
siblings. Raster names such as `optical` are arbitrary flat group names. See the
[model YAML](examples/model_spec.yaml) and [specification guide](spec/README.md).

## Ingestion and anchors

```python
from geosave_engine.workflow import ingest

raw_path = ingest(
    sources={
        "optical": {
            "url": "https://earth-search.aws.element84.com/v1/",
            "collection": "sentinel-2-l2a",
            "load": {
                "bands": ["red", "nir"],
                "with_properties": [{
                    "key": "view:sun_azimuth", "name": "sun_azimuth",
                    "dtype": "float32", "units": "degree",
                }],
            },
        },
    },
    anchor={
        "type": "coordinates", "latitude": -8.05, "longitude": 112.15,
        "shape": [512, 512], "resolution": 10, "timespan": "2025-01",
    },
    output="raw.zarr",
)
```

Each source uses a STAC API `url`, a `collection`, optional native `query`
settings and optional native `load` settings. The runtime builds `StacClient`,
`StacQuery` and `StacSourceConfig` directly. Model-independent ingestion loads
all supplied sources. Optional `spec="artifacts/model"` narrows sources and
bands to the model's requirements. Acquisition errors propagate.

| Anchor type | Required fields | Meaning |
| --- | --- | --- |
| `coordinates` | `latitude`, `longitude`, `shape`, `resolution` | Grid centred on a WGS84 point; shape is an integer or `[height, width]` |
| `geojson` | `path`, either `resolution` or `shape` | Grid covering a GeoJSON file's geometry |

Both forms accept `crs` and `timespan`. GeoJSON also accepts `pad`, in grid CRS
units. Native GeoAnchor chooses the grid and validates geometry; coordinates
inside GeoJSON follow its longitude/latitude order. A GeoJSON anchor covers the
geometry's extent. Pixel masking to a polygon remains an explicit operation.

```python
anchor = {
    "type": "geojson", "path": "aoi.geojson",
    "resolution": 10, "crs": "EPSG:32749",
    "timespan": ["2025-01-01", "2025-03-31"],
}
```

The [ingestion YAML](examples/ingest.yaml) is the same primitive argument mapping
and can be loaded with `yaml.safe_load` and passed to `ingest(**settings)`.

## Coordinates and attributes

A value changing with acquisition time belongs on a coordinate:
`sun_azimuth(time)`. `load.with_properties` uses odc-stac's native property loader
and grouping. `name` gives an explicit output name. GeoSave promotes these
auxiliary variables with native `set_coords`, retaining time labels, units and
lazy pixels. Without an alias, ODC supplies its normal property name. The
separate `item_properties` option records provenance snapshots in `StacMetadata`;
it does not create analysis coordinates.

For existing raster files, coordinates come from the raster. In Python they can
be assigned with the ordinary xarray API before saving:

```python
scene = scene.assign_coords(
    sun_azimuth=("time", sun_angles, {"units": "degree"}),
)
```

The attribute section validates metadata; it does not insert coordinate values.
Registered `models` and unregistered `foreign` attributes are siblings within
one root, variable or coordinate scope:

```yaml
attrs:
  data_vars:
    red:
      models:
        packing:
          required: [scale_factor]
      foreign:
        required: [provider_note]
  coords:
    sun_azimuth:
      models:
        coordinate:
          equals: {units: degree}
```

Coordinates follow native time windowing and spatial tiling. A `model_context`
callable receives each tile DataTree before pixels are converted and can encode
its sun angles, dates or other coordinates as model arguments. Public `predict`
accepts that callable's import path; ordinary `infer` accepts the callable itself.
Shared primitive model arguments are supplied through `context`.

## Reusable stages and sampling

```python
from geosave_engine.workflow import infer, postprocess, preprocess
from geosave_engine.workflow.io import open_rasters, write_stack
from geosave_engine.workflow.spec import ModelSpec

spec = ModelSpec.load("artifacts/model")
with open_rasters("raw.zarr") as raw:
    prepared = preprocess(raw, spec=spec)
    saved = write_stack(prepared, "prepared.zarr")
with open_rasters(saved) as prepared:
    logits = infer(prepared, model=loaded_model, settings=spec.inference)
    results = postprocess(logits, settings=spec.postprocessing)
```

`acquire`, `preprocess`, `infer` and `postprocess` remain ordinary native Python
APIs. Use `infer` directly for already prepared data. `predict` always prepares
raw inputs; it does not guess preparation state from metadata. Optional
`prepared_output` completes a Zarr write and reopens it before inference,
preventing repeated preparation for overlapping tiles.

```yaml
inference:
  inputs:
    image: {raster: reflectance, layout: TCHW, dtype: float32}
    elevation: {raster: terrain, layout: CHW, dtype: float32}
  time_window: {size: 4, stride: 2, tolerance: 30D, mode: strict}
  tiling:
    raster: reflectance
    tile_shape: [256, 256]
    overlap: 32
    window: hann
```

STAC loading retains time even for one acquisition. A spatial model explicitly
selects or aggregates it. `time_window` reuses native `window_stack`; static
modalities pass through. Only bound variables participate in temporal matching.
The tiling reference supplies the spatial grid, and bound rasters must match it.
`Tiles`, `RasterSamples` and PyTorch DataLoader form synchronized model batches.
These same sampling APIs can be used in training.

Each temporal window produces a separate raster with time coverage metadata.
`TileMerger` merges spatial logits before softmax or class thresholds. The
initial output contract is floating dense `[B, C, H, W]` at tile resolution.
Output paths are `window-0000.zarr`, etc., inside the requested directory.

## Execution and verification

Ingestion uses bounded source tasks, with four worker threads by default;
`ingest.with_options(task_runner=...)` uses ordinary Prefect behavior. Prediction
uses preprocessing, model loading, inference, postprocessing and write tasks.
Small operations execute inside their stage. A configured custom Prefect Task
uses its underlying function. Loaded models and lazy arrays are never automatic
persisted task results; actual completed raster writes are the checkpoints.

Owned files open with Dask chunks and close when their scope ends. Raw, prepared
and prediction writes use new local destinations and stage data before exposing
the result. Failed writes leave no partial checkpoint at the requested path.
Remote publishing and separate CPU/GPU deployment scheduling are not included.
Few-shot conditioning can use context; weight adaptation is not implemented.
Tile batches are bounded, while merged output rasters still occupy memory.

```bash
uv run pytest tests/workflow
```

Tests exercise local HTTP STAC documents and GeoTIFF assets, saved model reload,
GeoJSON/coordinate anchors, lazy file reading, native methods, temporal context,
spatial reconstruction, Prefect failures and native persistence. Prefect and
persistence tests need local sockets; no external catalog or model download is
required.
