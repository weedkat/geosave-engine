# Training Data Path Design

**Status:** implemented, uncommitted, 2026-10-03. Sections marked *agreed* were
reviewed one by one. Sections marked *decided during implementation* were
agreed in direction only; their details are mine and are open to change. The
plan and its rulings are in `plans/2026-10-03-training-data-path.md`.

## Purpose

The segmentation template cannot train today: `ml.segmentation.supervised.DataModule`
raises `NotImplementedError` when it builds a dataset, and the augmentation
support was removed by accident in `3b2cd23`. This spec makes the template train
end to end from prepared samples, with one definition of preprocessing shared by
training and prediction.

## Scope

In scope:

- A sample manifest that is a STAC table, built and read through `GeoVector`.
- `supervised.Dataset` and `supervised.DataModule` reading that manifest.
- A `transforms` stage in `model_spec.yaml` and the augmentation hook.

Out of scope, each later:

- `register` and `split` helpers. Filtering and splitting a manifest are ordinary
  pandas operations on `GeoVector.gdf` for now.
- LitData export, prediction and postprocessing, other head types,
  semi-supervised methods, and the map viewer.

## The pipeline

```text
disk sample (raw, one raster per layer)
  -> Dataset: cut frames along time (pairs the layers),
              run spec.preprocess on each frame,
              cut tiles in space                                xarray, lazy
  -> DataModule.on_after_batch_transfer:
       augmentations (training only, from train.yaml)           GPU tensors
       transforms    (every split, from model_spec.yaml)        GPU tensors
  -> Module.training_step / validation_step
```

Who owns what:

| Stage | Declared in | Runs in | When |
| --- | --- | --- | --- |
| Time cut into frames | `model_spec.yaml` `frames` | Dataset, per sample | reading from disk |
| Raster preprocessing: nodata, unpack, band order, stacking layers | `model_spec.yaml` `preprocessing` | Dataset, per frame | reading from disk |
| Space cut into tiles | `model_spec.yaml` `tiles` | Dataset | reading from disk |
| Augmentation | `train.yaml` `data.init_args.augmentations` | DataModule hook | training only |
| Tensor transforms, such as normalization | `model_spec.yaml` `transforms` | DataModule hook | every split, after augmentation |

- The Lightning module does not read the model spec. It receives model-ready
  batches. The DataModule is the only training component that takes the spec.
- Normalization lives in `transforms` only. It is never an augmentation, so a
  release always carries it, and it always runs after augmentation.
- Samples stay raw on disk. Changing `preprocessing` or `transforms` needs no
  re-preparation.
- Train and validation are separate manifest files, not a split column.

```yaml
# train.yaml
data:
  class_path: geosave_engine.ml.segmentation.supervised.DataModule
  init_args:
    spec: configs/model_spec.yaml
    train: data/train.parquet
    val: data/val.parquet
    target: label
    augmentations:
      - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
```

```yaml
# model_spec.yaml
frames: {length: 4, stride: 2, tolerance: 10D}
preprocessing: {...}
transforms:
  - {name: Normalize, init_args: {mean: [1105.0, 1355.0], std: [1809.0, 1757.0]}}
```

Both lists use the `{name, init_args}` form `ImageAugmenter` already parses.

## Cuts (agreed)

A sample is cut twice, and the two cuts are named as a pair: **frames** along
time, **tiles** in space. An *instant* stays the word for one time step.

```python
sample   = stack({name: gs.read_raster(a["href"]) for name, a in row["assets"].items()})
frames   = spec.frames(sample)                        # time: pairs layers on one joint axis
prepared = [spec.preprocess(frame) for frame in frames]
tiles    = Tiles(prepared, tile_shape, overlap=32)    # space
```

- The time cut is today's `window_stack` in `geodata/transform/time.py`. It lays
  groups that observe on different dates onto one joint axis and cuts it into
  runs of consecutive instants. A timeless group, such as a label or a DEM,
  joins every frame whole.
- The time cut runs before `preprocessing`. Combining layers such as S1 and S2
  into one image is only defined once they sit on the same instants.
- A cut is one raster in and many out, so it is not a `preprocessing` step. Its
  parameters live in their own `frames` section of the model spec, because
  prediction must cut the same frames the model was trained on.
- A single-date model omits `frames`, and no time cut happens.

"Window" is renamed because it also means the blending weights used when
stitching (`window="hann"`), which is the only meaning it keeps:

| Today | Becomes |
| --- | --- |
| `window(data, size, stride=)` | `frames(data, length, stride=)` |
| `window_stack(tree, size, tolerance=, stride=, mode=)` | `stack_frames(tree, length, tolerance=, stride=, mode=)` |
| `WindowMode` | `FrameMode` |
| `DroppedWindowsWarning` | `DroppedFramesWarning` |

`length` is the number of instants a frame holds, and `stride` the instants
between the starts of consecutive frames. Prithvi's own `num_frames` argument
counts instants, so it equals `frames.length`.

## Tiling and evaluation (agreed)

Tiling follows the model spec, like the time cut, so training, validation,
test, and prediction all cut the same way:

```yaml
# model_spec.yaml
tiles:
  size: 224
  overlap: 32
  mode: reflect     # how a tile reaching past the raster is filled
  window: hann      # how overlapping tiles are weighed when merged
```

The dataset yields every tile of every frame. Nothing is sampled at random.

- **Training** scores each tile on its own.
- **Validation and test** merge tile outputs back onto each raster first, then
  compute the loss and metrics on the whole raster. Every real pixel is scored
  once, padding is removed by the merge, and overlapping tiles are blended as a
  predict run blends them, so the metric describes the product a user gets.

The merge uses the existing `Tiles.merger(window=...)`. `TileMerger.merge()`
returns and releases each raster as soon as all its tiles have arrived, so
memory is bounded by the rasters in flight:

```python
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    model_inputs, index = batch
    logits = self(**model_inputs)
    merger.add(dict(zip(index.tolist(), logits.cpu().numpy())))
    for ordinal, merged in merger.merge().items():     # rasters now whole
        self.val_metrics.update(merged, target_of(ordinal))
```

The dataset owns the `Tiles`, so it supplies the merger and each raster's
target; the module still does not read the spec.

Limits to state in the plan:

- Tiles of one raster must reach one process. Multi-process (DDP) validation
  would split a raster across processes and never complete it.
- A validation run cut short, such as Lightning's sanity check, leaves rasters
  incomplete. Those are dropped at the end of the run, not scored in part.

## Model inputs (agreed)

`spec.preprocess` returns every named step, intermediate ones included. An
`inputs` section states which results the model takes, using the `!ref`
mechanism the spec already has:

```yaml
preprocessing:
  valid_pixels:
    call: geosave_engine.geodata.transform.nodata.to_nan
    kwargs: {data: !ref sentinel_2_l2a}
  image:
    call: geosave_engine.geodata.transform.packing.unpack
    kwargs: {data: !ref valid_pixels}
inputs:
  image: !ref image
```

- A key is the name the model chain takes. A value points at a preprocessing
  result or a declared raster, so step names and model input names are
  independent, and a raw raster needs no do-nothing step.
- The dataset turns exactly these entries into tensors.
- A reference that names nothing is refused when the spec loads, by the spec's
  own validator.

### Encoder context (agreed)

Some encoders take more than pixels: Prithvi TL takes `temporal_coords` and
`location_coords`, Clay takes `time` and `latlon`. An `inputs` value is
therefore either a reference or a call:

```yaml
inputs:
  image: !ref image                 # a raster: its pixels
  temporal_coords:                  # a call: run on each tile
    call: geosave_engine.model.encoder.prithvi.temporal_coords
    kwargs: {data: !ref image}
  location_coords:
    call: geosave_engine.model.encoder.prithvi.location_coords
    kwargs: {data: !ref image}
  time:
    call: geosave_engine.model.encoder.clay.time
    kwargs: {data: !ref image}
  latlon:
    call: geosave_engine.model.encoder.clay.latlon
    kwargs: {data: !ref image}
```

- `inputs` is evaluated on each tile, after the cut, so a location is the
  tile's own centre.
- Each encoder owns its functions, in its own module, named after the chain
  input they feed. They replace the `model_context` static methods and
  `TileDataset`'s `model_context=` argument.
- Which raster a context value is read from is the `!ref` in the call.
- A spec lists the inputs of every encoder it should work with. The model chain
  takes the ones it asks for and ignores the rest, so the encoder can be
  changed in `train.yaml` without editing the spec. No encoder input is
  renamed.
- A reference entry is pixels: it is tiled, augmented, and transformed. A call
  entry is passed to the model untouched.
- A required input the spec does not list fails on the first forward, by name:
  `KeyError: "PrithviTL.forward_pyramid: missing input 'temporal_coords'"`.

Limit: a listed call runs on every tile whether or not the selected encoder
takes it. `clay.time` refuses a tile with more than one time label, so a
multi-frame spec must not list it.

The module owns no data-shaped arguments. `in_channels` and `input_size` leave
`supervised.Module` and become ordinary `init_args` of whichever encoder the
chain names, so each encoder keeps its own argument names:

```yaml
model:
  init_args:
    model_chain:
      encoder:
        name: dinov3
        init_args: {in_channels: 4, input_size: 224}
```

There is no argument linking and no check between the spec and the encoder. A
mismatch surfaces from PyTorch on the first batch, during Lightning's sanity
check. The tile size is therefore written twice, in `tiles.size` and on the
encoder.

## Manifest (agreed)

### A manifest is a STAC table

Each row is one STAC item: one place, one time span, one grid, with named raster
layers. The file is GeoParquet following the stac-geoparquet specification, so
geopandas, pystac, and other STAC tools read it. `GeoVector` is its in-memory
form; there is no separate manifest class.

| Column | Meaning |
| --- | --- |
| `id` | sample identifier, unique within the table |
| `type`, `stac_version`, `stac_extensions`, `links` | STAC item columns |
| `geometry`, `bbox` | footprint in longitude/latitude |
| `datetime`, `start_datetime`, `end_datetime` | timezone-aware UTC |
| `proj:code`, `proj:shape`, `proj:transform` | the grid every layer shares |
| `assets` | `{layer name: {href, type, roles, bands}}` |
| any other column | a caller property, such as a class label or region |

### The record

`GeoVector.from_xarray` produces STAC item rows. Its own column names for the
same facts are replaced:

| Today | Becomes |
| --- | --- |
| `grid_crs` | `proj:code` |
| `grid_transform` | `proj:transform` |
| `grid_height`, `grid_width` | `proj:shape` |
| `path` | `assets.<layer>.href` |
| `variables` | `assets.<layer>.bands[].name` |

```python
sample.gs.to_cog("data/s1")           # s1/label.tif, s1/sentinel_2_l2a.tif

record = GeoVector.from_xarray(
    sample,
    assets={"label": "s1/label.tif",
            "sentinel_2_l2a": "s1/sentinel_2_l2a.tif"},
    land_cover="forest",
)
catalog = GeoVector.concat([catalog, record])
catalog.to_geoparquet("data/manifest.parquet")
```

- A relative href is read as relative to the manifest. An absolute href below
  the manifest's folder is stored relative to it; any other href is kept.

- One asset per layer. An asset key is a layer name and must name a group of
  the stack. Each asset points at one raster.
- `from_xarray` reads what the raster knows: footprint, bounding box, times,
  `proj:*`, each layer's band names, and the media type from the suffix.
- The caller supplies the asset paths, optional `roles`, and custom properties.
- `id` is optional and defaults to `data.gs.anchor.stem`, for example
  `2.7415W_5.6550N_5.1kmx5.1km_20181226_10m`. It is the key for
  `GeoVector.upsert(records, on="id")` and for joining caller metadata.
- A raster with no time is refused unless the caller passes `datetime=`.
- `datetime` is null unless the caller passes it; `start_datetime` and
  `end_datetime` carry the raster's timespan, which covers whole days.
- The grid CRS is `proj:code` when it has an EPSG code, else `proj:wkt2`.

### Opening a row

Loading uses GeoSave's own readers. Every asset opens with `read_raster`, which
picks the GeoTIFF, Zarr, or NetCDF reader from the suffix, and the layers join
with `stack`:

```python
sample = stack({name: gs.read_raster(asset["href"]) for name, asset in row["assets"].items()})
```

Layers may mix formats, such as a `.tif` label beside a `.zarr` time series.

Limits accepted for now:

- A multi-group Zarr stack is not a sample format. `read_raster` refuses a
  group inside a store, so each layer is saved as its own raster.
- A time series saved as a tree of GeoTIFFs is not one asset. A time-series
  layer is saved as Zarr.

### Writing and reading

A table with an `assets` column is a STAC table. Ordinary vectors keep today's
behaviour.

Writing a STAC table with `GeoVector.to_geoparquet`:

- Asset hrefs below the file's folder are stored relative to it, using the
  existing `stored_asset_path`. Other paths and remote URLs are kept as written.
- A null or repeated `id` is refused, naming the duplicates.
- A CRS other than longitude/latitude is refused, telling the caller to call
  `to_crs("EPSG:4326")`.
- The bounding box column is always written.
- The `stac-geoparquet` file metadata key is added.

Reading with `gs.read_vector`:

- Asset hrefs are expanded against the file's location with the existing
  `resolve_asset_path`, so in memory an href is directly openable.
- An asset a row does not have is absent from that row's mapping. Parquet
  stores it as a null entry, which the reader drops.

### What this replaces

- The `path` column handling in the GeoParquet reader and writer is removed, as
  are `from_xarray`'s `path=` argument and `variables` field.
- `workflow/tasks/manifest.py` `write_manifest` builds one record per sample
  with `GeoVector.from_xarray(..., assets=..., id=<sample ID>)`. The manifest's
  `format` column and `sample_path` are gone.
- `workflow/tasks/sample.py` keeps `write_sample` and `open_sample`: they hold
  the atomic publication and the completed-sample check that `to_cog` alone
  does not. A sample is a folder of one raster per layer in either format, so a
  Zarr sample is `<sample>/<layer>.zarr`, no longer one multi-group store.
- `examples/data/dw_imagery/manifest.parquet` is still in the old shape. It is
  a data file with uncommitted changes, so it was left for its owner.

### Dependency

`stac-geoparquet>=0.8.2`, added 2026-10-03. `GeoVector` writes the table; the
library hands rows to STAC tools as `pystac` items and is used in tests to prove
conformance.

## Verified so far

Each with a throwaway script on the three example samples:

- A table written by the stac-geoparquet library reads through `gpd.read_parquet`
  and `gs.read_vector`, passes `GeoVector` validation, and supports `query`,
  pandas filters, and `concat`.
- A table built with today's `GeoVector.from_xarray` plus the item columns and
  written with `to_geoparquet(write_covering_bbox=True)` reads back through the
  STAC library; every row validates against the STAC 1.1 item schema once
  `type` and `stac_version` are present. A row written without a label asset
  reads back with only its image asset.
- Without the bounding box the STAC reader refuses the file. geopandas drops
  that column unless the option is set.
- Adding the `stac-geoparquet` metadata key with pyarrow leaves the file
  readable by geopandas and the STAC reader.
- Per-layer GeoTIFF assets open with `read_raster` and join with `stack`.
- Merge then score: two 510x510 samples cut into 32 tiles (224, overlap 32),
  fed to `TileMerger` in shuffled batches with a `hann` window, rebuilt both
  rasters at 510x510. Scoring the merged output touched 520,200 pixels, exactly
  2 x 510 x 510, and nothing was left pending.
- `Tiles` cuts a two-layer stack by one footprint: a 510x510 sample gives nine
  224x224 tiles, each holding both layers.
- A `Normalize` step declared in YAML and built by `ImageAugmenter` runs after a
  flip, normalizes the image, and leaves the mask untouched.

Not yet verified: shortening hrefs inside nested `assets` cells, writing the
metadata key in the same pass, the id and CRS checks, and a row mixing `.tif`
and `.zarr` assets.

## Defects found on the way

Still open; the first two sit in attrs code with uncommitted work:

1. `read_tree` fails on a written COG leaf with `KeyError: 'stac_items'` at
   `geodata/attrs/models/stac.py:101` (`cls.field_keys["stac_items"]`).
2. `stack.gs.to_netcdf` fails with `TypeError: illegal data type for attribute
   'stac_items'` when the stack carries STAC metadata.
3. `read_stack` cannot read what `stack.gs.to_cog` writes, and `read_raster`
   refuses a COG tree directory.
4. A raster opened with `read_raster(chunks=...)` is not released by
   `.close()`; its files close only once nothing refers to the Dataset.

Fixed here:

5. `Tiles` with `reflect` or `wrap` failed on a lazy raster whenever a tile had
   to invent more pixels than it held (`conflicting sizes for dimension`):
   dask's padding stops after one pass. Positions are now laid out by numpy.
6. The template's model chain did not build, because encoders declared
   `tuple[list, list]` where decoders take `list[torch.Tensor]`. A test now
   builds each registered encoder with a decoder and a head.
7. Kornia rejects a `(B, T, C, H, W)` batch. The DataModule folds time away
   around each Kornia call.

Found while testing worker processes, and designed around:

8. After a fork, a child's normal exit closes the raster handles it inherited,
   which breaks the parent's own handles (`RasterioIOError: Read failed`).
9. A parent that has already read pixels makes a forked child hang on its
   first read.

## Dataset (decided during implementation)

`supervised.Dataset` in `ml/segmentation/supervised/data.py`:

```python
dataset = Dataset(read_vector("data/train.parquet"), spec, target="label")
model_inputs, target, index = dataset[0]
```

- At construction it opens every row lazily, cuts frames, runs
  `spec.preprocess` on each frame, and cuts one `Tiles` over all of them. One
  `Tiles` gives the global tile numbering, `locate`, and the merger.
- A sample is `(model_inputs, target, index)` in every split.
- The target is the raw layer named by `target`, tiled with the inputs. A
  target pixel the raster does not have becomes `ignore_index`.
- Building it opens every raster once to count tiles, then lets them go. Each
  process, the parent included, opens its own rasters on first use, and a
  pickled dataset carries none.
- `DataModule` starts workers with `forkserver`, so a worker inherits neither
  open rasters nor reader threads (defects 8 and 9), and keeps them alive
  across epochs.

Limits: every sample is opened once to count and once per worker, which is
slow for a very large manifest. Preprocessing must keep the sample's grid,
because inputs and target are cut by one footprint. A loader that forks and is
started after this process has read pixels can hang; use `forkserver`.

## DataModule (decided during implementation)

```python
DataModule(spec, train, val, *, test=None, target="label", augmentations=None,
           ignore_index=255, batch_size=8, num_workers=4)
```

`on_after_batch_transfer` runs on the device:

1. Training only: `augmentations` on the pixel inputs and the target together.
2. Every split: the spec's `transforms` on each pixel input.
3. Every split: NaN pixels become 0, which is the mean once normalized.

A `(B, T, C, H, W)` input is folded to `(B, T*C, H, W)` for augmentation, so
every instant gets the same geometry, and to `(B*T, C, H, W)` for transforms,
so per-band statistics apply to each instant.

`transforms` is keyed by input name, because a model may take several rasters:

```yaml
transforms:
  image:
    - {name: Normalize, init_args: {mean: [...], std: [...]}}
```

`ImageAugmenter` builds both pipelines and keeps its name.

## Module (decided during implementation)

`supervised.Module` drops `in_channels` and `input_size`. A batch is
`(model_inputs, target, index)` in every split. Validation and test merge
logits and target tiles with two mergers from the loader's dataset, then score
each raster as it completes. A validation epoch that completes no raster logs
no `val_loss` and warns, naming `limit_val_batches`.

A pixel an augmentation invents, such as the border a shift leaves, becomes
`ignore_index` in the target.

## Template and tests

`augmentations` restored in the template `train.yaml`; the template spec gains
`tiles`, `inputs` for every built-in encoder, and `transforms` with
Sentinel-2 reflectance statistics from `geodata/sensors/sensors.yaml`.

A slow test generates the workspace, lists the three example samples in
manifests, and fits it: two training batches, then a full validation that
scores three 510x510 rasters whole. The example rasters hold digital numbers,
not reflectance, so that test swaps in statistics in their units.
