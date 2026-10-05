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
  composing new flows. Sample writing and label reading remain in their
  focused task modules.

Raster requirements, acquisition recipes, and preprocessing stay on
`ModelSpec`, which is shared by workflows, training, release, and serving.

| Command | Purpose | Input | Output |
| --- | --- | --- | --- |
| `ingest` | Load every model raster on one explicit spatial and temporal target. | Anchor JSON and model spec. | One sample directory holding one raster per model raster, and optionally its row in a catalog. |
| `prepare-dense-data` | Match imagery to label rasters for training. | Label table or label directory, and model spec. | One directory per sample, holding one raster per layer, and one STAC GeoParquet manifest. |

## Ingest one sample

```bash
geosave workflow ingest \
  --anchor '{"kind":"raster","path":"data/reference.tif"}' \
  --output data/scenes/s1 \
  --spec model_spec.yaml
```

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--anchor JSON` | Yes | — | Spatial and temporal anchor. See the supported shapes below. |
| `--output PATH` | Yes | — | New sample directory, which takes one raster per model raster. |
| `--spec PATH` | Yes | — | `model_spec.yaml` file or model artifact directory. |
| `--format [geotiff\|zarr]` | No | `zarr` | Format of each raster. GeoTIFF holds one instant; Zarr holds a time series. |
| `--write-options JSON` | No | `{}` | Encoding options passed to the native writer. |
| `--catalog PATH` | No | — | GeoParquet STAC table recording what was written. The sample's row, named after the output directory, is added or replaces the row with that name. |
| `--help` | No | — | Show command help. |

The sample opens with `read_stack("data/scenes/s1")`, one group per model
raster. With `--catalog`, `read_vector("catalog.parquet").set_index("id").loc["s1"].gs.to_xarray()`
opens it by name.

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
  --output data/scenes/s1 \
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
| `--labels PATH` | Yes | — | GeoParquet label table, or a root containing label rasters. |
| `--output PATH` | Yes | — | Directory for prepared samples and `manifest.parquet`. |
| `--spec PATH` | Yes | — | `model_spec.yaml` file or model artifact directory. |
| `--pattern TEXT` | No | `**/*.tif` | Label glob relative to `--labels`, when it is a directory. |
| `--max-concurrency INTEGER` | No | `1` | Maximum simultaneous complete sample ingestions. Must be at least one. |
| `--format [geotiff\|zarr]` | No | `geotiff` | Prepared sample representation. |
| `--write-options JSON` | No | `{}` | Encoding options passed to the native writer. |
| `--help` | No | — | Show command help. |

A label table is a STAC GeoParquet table with one row per label: a unique
`id`, an asset named `label` pointing at the label raster, a time, and any
columns of your own. A directory of label rasters is indexed into such a table
for you. Build one from a directory, add your columns, and write it back:

```python
from geosave_engine.workflow.tasks.labels import read_labels

labels = read_labels("data/labels")
labels["quality"] = survey_quality
labels.gs.to_geoparquet("data/labels.parquet")
```

For a directory, sample IDs mirror the label path relative to `--labels`, with
only the final file suffix removed. For example, `data/labels/train/region/tile.v1.tif`
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
are flat sibling rasters, one per layer, which keeps discovery simple without
combining unrelated rasters into one physical file. GeoTIFF samples accept a
raster with no time dimension or one time step. Use Zarr when a named raster
contains multiple time steps:

```bash
geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml \
  --format zarr
```

That writes the same directory with one Zarr store per layer,
`data/prepared/train/region/tile.v1/label.zarr` and
`.../sentinel_2_l2a.zarr`. The source tree is preserved for either format, so
directories such as `train`, `val`, and `test` remain useful for dataset
discovery.

`manifest.parquet` is a STAC GeoParquet table: each row is one sample, with its
`id`, footprint, timespan, grid (`proj:code`, `proj:shape`, `proj:transform`),
and one asset per layer. Asset hrefs are stored relative to the manifest, so
the prepared directory can be moved as a whole. The table reads as a
GeoDataFrame, and a row opens as a lazy stack through its `gs` accessor:

```python
from geosave_engine.geodata import read_vector

manifest = read_vector("data/prepared/manifest.parquet")
sample = manifest.iloc[0].gs.to_xarray()
```

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

Every column of a label row that is not one of the manifest's own is copied
onto its sample row. The manifest's own columns are `id`, `type`,
`stac_version`, `stac_extensions`, `links`, `datetime`, `start_datetime`,
`end_datetime`, the `proj:` grid fields, `assets`, `sources`, `bbox`, and
`geometry`. `sources` lists the provider items each sample was loaded from.

The manifest is written as each sample finishes. A run that fails leaves the
finished samples recorded, and the next run prepares only the rest: a recorded
sample is reused without opening its files, provided its layers are the label
and the model spec's rasters. Splits are separate manifest files, cut from the
finished one:

```python
manifest = read_vector("data/prepared/manifest.parquet")
manifest[manifest.region == "north"].gs.to_geoparquet("data/prepared/train.parquet")
```

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
`attrs` requirements are optional and validation stays lazy. A single-date
raster, such as one read from a GeoTIFF, carries its `time` as a scalar; where
`dims` names `time` it is given a length-one axis.

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
from geosave_engine.model.spec import ModelSpec

model = ModelSpec.load("model_spec.yaml")
prepared = model.preprocess({"sentinel_2_l2a": raster})
```

## Failure and resume behavior

Each sample is published atomically: GeoSave writes to a temporary sibling
directory and moves it into place only when all writes succeed. After the first
sample failure, the flow submits no later labels. Work already in flight
finishes, and every sample that completes is recorded before the failure is
reported.

The manifest is rewritten as each sample finishes, so a failed run leaves a
manifest of the samples it completed. Rerunning the same command reuses every
recorded sample whose layers and bands match the model spec, registers a
complete sample directory that has no row yet, and prepares the rest. A
recorded sample that does not match the spec stops the run. When the run
completes, the manifest holds exactly the current labels, in label order, with
caller columns taken from the label table as it is now.

`ingest` checks its `--catalog` before loading anything, and refuses an output
directory that already exists. Its catalog row is named after the output
directory alone, so two scenes written to `a/s1` and `b/s1` share one row.

## Python equivalent

The commands are thin wrappers around the two public flows:

```python
from geosave_engine.workflow.flows import ingest, prepare_dense_data

scene = ingest(
    anchor={"kind": "raster", "path": "data/reference.tif"},
    output="data/scenes/s1",
    spec="model_spec.yaml",
    catalog="data/scenes/catalog.parquet",
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
