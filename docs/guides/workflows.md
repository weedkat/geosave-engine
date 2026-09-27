# Run workflows

GeoSave exposes complete, durable jobs under `geosave workflow`. Commands run
locally, wait for Prefect to finish, and print the output path only after a
successful run.

| Command | Purpose | Input | Output |
| --- | --- | --- | --- |
| `ingest` | Load every model source on one explicit spatial and temporal anchor. | Anchor JSON, model spec, and optional source settings. | One Zarr raster stack. |
| `prepare-dense-data` | Match imagery to local label rasters for training. | Label directory, model spec, and optional source settings. | Sample Zarr stores and one GeoParquet manifest. |

## Ingest one raster stack

```bash
geosave workflow ingest \
  --anchor '{"kind":"raster","path":"data/reference.tif"}' \
  --output data/raw.zarr \
  --spec model_spec.yaml
```

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--anchor JSON` | Yes | — | Spatial and temporal anchor. See the supported shapes below. |
| `--output PATH` | Yes | — | Destination Zarr store. |
| `--spec PATH` | Yes | — | `model_spec.yaml` file or model artifact directory. |
| `--sources JSON` | No | Model defaults | Query and loading settings keyed by model source name. |
| `--help` | No | — | Show command help. |

The anchor may copy an existing raster's exact grid and time:

```json
{"kind":"raster","path":"data/reference.tif"}
```

It may cover a local GeoJSON file. Supply exactly one of `shape` or
`resolution`; `crs`, `timespan`, and `pad` are optional.

```json
{
  "kind": "geojson",
  "path": "data/area.geojson",
  "resolution": 10,
  "crs": "EPSG:32633",
  "timespan": "2025-01",
  "pad": 100
}
```

Or it may build a grid around a WGS84 coordinate. `shape` can be one integer
or `[height, width]`; `crs` and `timespan` are optional.

```json
{
  "kind": "coordinates",
  "latitude": 45.0,
  "longitude": 12.0,
  "shape": [1024, 1024],
  "resolution": 10,
  "crs": "EPSG:32633",
  "timespan": ["2025-01-01", "2025-01-31"]
}
```

Pass the JSON as one quoted shell argument:

```bash
geosave workflow ingest \
  --anchor '{"kind":"coordinates","latitude":45,"longitude":12,"shape":1024,"resolution":10,"timespan":"2025-01"}' \
  --output data/raw.zarr \
  --spec model_spec.yaml
```

## Prepare dense training data

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml
```

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--labels PATH` | Yes | — | Root containing label rasters. |
| `--output PATH` | Yes | — | Directory for sample Zarrs and `manifest.parquet`. |
| `--spec PATH` | Yes | — | `model_spec.yaml` file or model artifact directory. |
| `--pattern TEXT` | No | `**/*.tif` | Label glob relative to `--labels`. |
| `--sources JSON` | No | Model defaults | Query and loading settings keyed by model source name. |
| `--max-concurrency INTEGER` | No | `1` | Maximum simultaneous complete sample ingestions. Must be at least one. |
| `--help` | No | — | Show command help. |

Sample IDs are label paths relative to `--labels`. For example,
`data/labels/train/a.tif` becomes sample ID `train/a.tif` and is written to
`data/prepared/samples/train/a.zarr`.

The default `--max-concurrency 1` is the safe setting: only one complete sample
ingestion runs at a time, from STAC search through Zarr persistence. This is a
concurrency limit, not an HTTP requests-per-second rate. Opt into parallel
sample ingestion only after confirming that the STAC service, raster sources,
storage, and local resources can sustain it:

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml \
  --max-concurrency 2
```

## Source parameters

When `--sources` is omitted, GeoSave creates default query and loading settings
for every source declared in `model_spec.yaml`. To override them, provide one
JSON object whose keys exactly match the model's source names:

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml \
  --sources '{"sentinel_2_l2a":{"query":{"max_items":8},"load":{"chunks":{"x":512,"y":512}}}}'
```

Each source accepts:

- `query`: STAC search parameters such as `bbox`, `intersects`, `datetime`,
  `filter`, `max_items`, `limit`, `ids`, and `sortby`;
- `load`: raster loading parameters accepted by GeoSave's STAC loader, such as
  `chunks`, `groupby`, `resampling`, and error behavior.

Unknown fields, malformed JSON, missing source names, and extra source names
are rejected before raster work starts. Collection identity, required
variables, endpoints, and raster constraints remain model-owned in
`model_spec.yaml`; they are not repeated in the command.

## Failure and resume behavior

Each sample is published atomically: GeoSave writes to a temporary sibling
store and moves it into place only when all chunks succeed. After the first
sample failure, the flow submits no later labels. Work already in flight may
finish safely, and its completed samples remain available for the next run.

The manifest is written only after every requested sample succeeds. A failed
run therefore does not replace an existing manifest with a partial one.
Rerunning the same command validates and reuses compatible completed samples.

## Python equivalent

The commands are thin wrappers around the two public flows:

```python
from geosave_engine.workflow.flows import ingest, prepare_dense_data

stack = ingest(
    anchor={"kind": "raster", "path": "data/reference.tif"},
    output="data/raw.zarr",
    spec="model_spec.yaml",
)

manifest = prepare_dense_data(
    labels="data/labels",
    output="data/prepared",
    spec="model_spec.yaml",
    max_concurrency=1,
)
```

Both calls are synchronous and return the completed output path. Supply
`sources={...}` with the same nested dictionaries accepted by `--sources` when
runtime query or loading overrides are needed.
