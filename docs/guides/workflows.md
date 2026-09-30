# Run workflows

GeoSave exposes complete, durable jobs under `geosave workflow`. Commands run
locally, wait for Prefect to finish, and print the output path only after a
successful run.

The Python package follows the same separation of concerns:

- `geosave_engine.workflow.configs` defines serializable runtime inputs such
  as anchor payloads.
- `geosave_engine.workflow.flows` exposes independently runnable Prefect
  workflows.
- `geosave_engine.workflow.tasks` exposes reusable Prefect work units for
  composing new flows. Persistence and manifest helpers remain in their
  focused task modules.

Raster requirements, acquisition recipes, and preprocessing stay on
`ModelSpec`, which is shared by workflows, training, release, and serving.

| Command | Purpose | Input | Output |
| --- | --- | --- | --- |
| `ingest` | Load every model raster on one explicit spatial and temporal target. | Anchor JSON and model spec. | One Zarr raster stack. |
| `prepare-dense-data` | Match imagery to local label rasters for training. | Label directory and model spec. | GeoTIFF sample directories and one GeoParquet manifest by default. |

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
| `--output PATH` | Yes | — | Directory for prepared samples and `manifest.parquet`. |
| `--spec PATH` | Yes | — | `model_spec.yaml` file or model artifact directory. |
| `--pattern TEXT` | No | `**/*.tif` | Label glob relative to `--labels`. |
| `--max-concurrency INTEGER` | No | `1` | Maximum simultaneous complete sample ingestions. Must be at least one. |
| `--format [geotiff\|zarr]` | No | `geotiff` | Prepared sample representation. |
| `--write-options JSON` | No | `{}` | Encoding options passed to the native writer. |
| `--metadata PATH` | No | — | CSV, TSV, Parquet, or XLSX rows keyed by relative `label_path`. |
| `--help` | No | — | Show command help. |

Sample IDs mirror the label path relative to `--labels`, with only the final
file suffix removed. For example, `data/labels/train/region/tile.v1.tif`
becomes sample ID `train/region/tile.v1` and the default output is:

```text
data/prepared/
├── manifest.parquet
└── train/
    └── region/
        └── tile.v1/
            ├── label.tif
            └── sentinel_2_l2a.tif
```

Each sample directory is one logical stack. The label and every named scene
are flat sibling multiband COGs, which keeps ordinary GeoTIFF discovery simple
without combining unrelated rasters into one physical file. GeoTIFF samples
accept a raster with no time dimension or one time step. Use Zarr when a named
raster contains multiple time steps:

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml \
  --format zarr
```

That writes the same example to
`data/prepared/train/region/tile.v1.zarr`. The source tree is preserved for
either format, so directories such as `train`, `val`, and `test` remain useful
for dataset discovery.

Writer-specific encoding settings are job inputs rather than model settings.
Pass them as a JSON object, for example:

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml \
  --write-options '{"compress":"ZSTD","blocksize":256}'
```

GeoSave continues to own the sample layout, eager publication, overwrite
policy, and band grouping; those settings cannot be supplied through
`--write-options`.

Optional metadata must contain exactly one row per discovered label and a
`label_path` column relative to the metadata table. Other columns are copied
to `manifest.parquet` in table order. Manifest-owned columns such as `path`,
`format`, grid fields, timestamps, and `geometry` cannot be overridden.

The default `--max-concurrency 1` is the safe setting: only one complete sample
ingestion runs at a time, from STAC search through sample persistence. This is a
concurrency limit, not an HTTP requests-per-second rate. Opt into parallel
sample ingestion only after confirming that the STAC service, remote assets,
storage, and local resources can sustain it:

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml \
  --max-concurrency 2
```

## Raster requirements and STAC recipes

The model spec owns both the raster it accepts and, when a workflow must fetch
that raster, its STAC recipe:

```yaml
schema_version: 2
rasters:
  sentinel_2_l2a:
    variables: [B04, B08]
    coordinates: [x, y]
    dims: [time, y, x]
    dtypes: [uint16]
    require_crs: true
    attrs:
      data_vars:
        "*":
          models:
            packing:
              required: [scale_factor]
    stac:
      collection: sentinel-2-l2a
      endpoints:
        - https://planetarycomputer.microsoft.com/api/stac/v1/
      query:
        datetime: [2025-01-01, 2025-01-31]
        max_items: 8
      load:
        bands: [B04, B08]
        chunks: {x: 512, y: 512}
```

`variables` selects named data variables in order. Use `channels` instead when
the model consumes the first N channels positionally. `coordinates` requires
coordinate arrays to exist without constraining their values. Grid, dtype, and
`attrs` requirements are optional and validation stays lazy.

The nested `stac` block is required by the `ingest` and
`prepare-dense-data` flows. It records the collection, fallback endpoints,
search parameters, and native raster loading settings in one portable model
artifact. Named `load.bands`, when present, must exactly match `variables` in
the same order.

Explicit recipe selectors take priority. `bbox`, `intersects`, and `datetime`
are preserved; the target supplies only an omitted spatial extent or time
window. An `ids` query receives no target-derived search selectors. The target
still defines the exact output grid used to load pixels.

If a raster already exists, no STAC recipe is needed. Supplying the native
`xarray.Dataset` to preprocessing applies the same selection and validation
before any preprocessing call:

```python
from geosave_engine.model_spec import ModelSpec

model = ModelSpec.load("model_spec.yaml")
prepared = model.preprocess({"sentinel_2_l2a": raster})
```

## Failure and resume behavior

Each sample is published atomically: GeoSave writes to a temporary sibling
directory or store and moves it into place only when all writes succeed. After
the first sample failure, the flow submits no later labels. Work already in
flight may finish safely, and its completed samples remain available for the
next run.

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
    format="geotiff",
    write_options={"compress": "ZSTD", "blocksize": 256},
)
```

Both calls are synchronous and return the completed output path. Raster
requirements and STAC acquisition settings come from `model_spec.yaml`.

When validating an anchor payload outside the built-in flow, import
`AnchorConfig` from `geosave_engine.workflow.configs`. When composing a custom
Prefect flow that prepares individual samples, import `prepare_dense_sample`
from `geosave_engine.workflow.tasks`. These lower-level interfaces support
composition; application entry points should generally call the complete
flows above.
