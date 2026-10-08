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
| `ingest` | Load every model raster on one explicit spatial and temporal target. | Anchor JSON and model spec. | One sample directory holding one raster per model raster. |
| `prepare-dense-data` | Match imagery to label rasters for training. | Label table or label directory, and model spec. | Public skeleton; currently raises `NotImplementedError`. |

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
| `--help` | No | — | Show command help. |

The sample opens with `read_stack("data/scenes/s1")`, one group per model
raster. To list the sample in a catalog, build one Item per group with
`stac.create_stack_items(paths, name="s1")`, where `paths` maps each group to its
saved files, and store them with `stac.table.write`.

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

`prepare_dense_data` and `prepare_dense_sample` retain their public signatures
as skeletons and raise `NotImplementedError`. Their previous catalog validation,
reconstruction and resume machinery has been removed. Directory label indexing
(`read_labels`) and training from catalog rows work; pixel writing and model
acquisition remain available independently.

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

## Python equivalent

```python
from geosave_engine.workflow.flows import ingest

scene = ingest(
    anchor={"kind": "raster", "path": "data/reference.tif"},
    output="data/scenes/s1",
    spec="model_spec.yaml",
)
```

Ingestion is synchronous and returns the completed output directory. Existing
outputs are refused. Raster requirements and STAC acquisition settings come
from `model_spec.yaml`. Import `AnchorConfig` from
`geosave_engine.workflow.configs` to validate anchor payloads independently.
