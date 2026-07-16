# GeoDataset and model definition

Reference for turning ingested `.geostack` folders into training batches,
and for the standardized `SemanticSegmentationTask`/`DataModule` pair. For
the ordered how-to (pick a path, train, evaluate), see
[workflow.md](../guide/workflow.md#5-define-the-model-).

## From ingested data to tensors: GeoDataset

`geosave_engine.geodata.datasets.GeoDataset` reads a directory
[save_dataset](pipeline.md#saving-to-disk) produced:

```python
from geosave_engine.geodata.datasets import GeoDataset

ds = GeoDataset("data/train")
```

Discovery is `root.rglob("*.geostack")` — every anchor folder, at any
nesting depth (see
[geotile.md](geotile.md#the-geostack-folder-convention) for why the
`.geostack` suffix exists and how nested grouping works). `__getitem__`
delegates straight to `GeoStack.to_tensor`:

```python
{
    "sentinel_2_l1c": torch.Tensor,   # [C, H, W]
    "dynamicworld": torch.Tensor,     # [1, H, W]
    "context": {...},                 # see below
}
```

| Constructor arg | Default | Purpose |
| --- | --- | --- |
| `root` | — | Directory to `rglob` for `.geostack` folders. |
| `required_layers` | `None` (all layers) | Only include anchors whose `.geostack` folder has every one of these layer names — a folder missing one is silently excluded. |
| `sel_bands` | `None` (all bands) | `{layer: [band, ...]}` — subset bands per layer. |
| `dtype_override` | `None` (saved dtype) | `{layer: torch.dtype}` — cast a layer's tensor, e.g. a saved `uint8` mask to `bool`. |
| `context_fn` | `None` (no `"context"` key) | `Callable[[dict[str, GeoTile]], dict[str, Any]]` — per-sample metadata, see below. |

**`layers` property** — the layer names present in the first sample
(`list[str]`), useful for sanity-checking a dataset without indexing into
it.

**Building your own Dataset:** subclass `GeoDataset` when you need
per-sample logic beyond what it gives you — override `context_fn` behavior
or wrap it, for instance. For streaming inference over fresh data without
ingesting to disk first, there's [stream_ingest()](pipeline.md#streaming-without-saving).

## What is context

Pass a `context_fn` to attach per-sample geo metadata, pulled from one
reference tile per sample (whichever layer happens to be first — they share
geobox/datetime by construction, so it doesn't matter which). The workspace
`Pipeline.context` (see
[Anatomy of a GeoPipeline](pipeline.md#anatomy-of-a-geopipeline)) is a
ready-made one:

| Field | From |
| --- | --- |
| `crs` | `tile.crs` |
| `transform` | `tile.affine` |
| `coordinate` | `tile.centroid` |
| `time` | `tile.start.timetuple().tm_yday` (day of year) |
| `datetime` | `tile.start.isoformat()` |
| `bbox_wgs84` | `tile.wgs84_bbox` |
| `stac_item_ids` | `[i.id for i in tile.stac]` |

Leave `context_fn` unset (the default) and every sample's `"context"` key is
just `{}` — no cost, nothing extracted.

## The zarr format

Ingestion writes zarr, not GeoTIFF, because a layer can carry a time
dimension (`(time, band, y, x)`) natively — a plain GeoTIFF can't.
`GeoTile.to_zarr`/`from_zarr` (see [geotile.md](geotile.md)) round-trip the
array plus datetime, metadata, and polygon footprint as store attributes —
no guessing from the path.

`GeoDataset`'s discovery pass opens every store with `load_data=False` —
header-only, geobox and datetime read from attrs without touching pixels.
Scanning a large `data/train/` directory to build the sample index is cheap
for exactly this reason. Pixels are only materialized when `__getitem__`
calls `tile.to_tensor()`, which clips to the tile's bbox and reads — so the
cost of one epoch is proportional to what the `DataLoader` actually visits,
not to how many `.zarr` stores exist on disk.

## stack_samples

The `DataLoader`'s `collate_fn`. Turns a list of per-sample dicts into one
batched dict, recursively, by value type:

```python
def stack_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    # tensor values  -> torch.stack(values)      (one stacked Tensor)
    # dict values     -> stack_samples(values)     (recurse — this is what batches "context")
    # anything else  -> list(values)               (gathered as-is)
```

Because `GeoDataset.__getitem__` keys samples by raw layer name, the batch
you get back has one tensor key per ingested layer, unchanged:

```python
{"sentinel_2_l1c": Tensor[B, C, H, W], "dynamicworld": Tensor[B, 1, H, W], "context": {...}}
```

`"context"` is a dict per sample, so it recurses — you get
`{"crs": [B values], "coordinate": [B values], ...}`, list-collated per
field since none of those are tensors.

## SemanticSegmentationTask

`geosave_engine.ml.tasks.SemanticSegmentationTask` — a standardized,
config-only semantic segmentation `LightningModule`. Owns model
construction, forward pass, sliding-window inference, postprocessing, and a
generic supervised training loop.

Model construction is stage-based — `stages` maps stage name to a registry
key (or an `nn.Module` class), built in order:

```yaml
stages:
  encoder: dinov3
  decoder: dpt
  head: dense
```

The first stage receives `in_channels`/`input_size`, the last receives
`num_classes` — both by position, not by a fixed stage name, so this works
whether `stages` has one entry (a monolith model) or several (a chain).

**`in_channels`/`num_classes` are not passed directly** — they're derived
from `band_map`/`class_map` (`len(band_map)`, `len(class_map)`). Both are
required, and their keys must be dense `0..n-1` — a hand-typed gap or
duplicate raises `ValueError` immediately rather than silently misaligning
class or channel indices.

| Arg | Required | Purpose |
| --- | --- | --- |
| `stages` | No (defaults to dinov3/dpt/dense) | Stage name → registry key, in build order. |
| `class_map` | **Yes** | `{class_id: class_name}`, dense from 0. `num_classes` is `len(class_map)`. |
| `band_map` | **Yes** | `{channel_idx: band_name}`, dense from 0. `in_channels` is `len(band_map)`. |
| `color_map` | No | `{class_id: hex_color}` — prediction visualization only; skipped with a warning if unset. |
| `image_key` / `label_key` / `mask_key` | No (`image`/`label`/`mask`) | Batch keys — point these at your `GeoDataset`'s raw layer names instead of requiring a renaming step. |
| `ignore_index` | No (`255`) | Class index excluded from loss and metrics. |
| `input_size` | No (`224`) | Spatial patch size for sliding-window inference. |
| `loss` / `optimizer` / `scheduler` | No | Registry keys — see `ml/core/factory.py` for what's registered (`CELoss`/`OHEMLoss`; `AdamW`/`Adam`/`SGD`/`RMSprop`/`Adagrad`; `CosineAnnealingLR`/`LRScheduler`). |
| `config` | No | Stage name → that stage's own constructor kwargs, plus `optimizer`/`scheduler` sub-keys for their kwargs. |
| `augmentations` | No | Kornia augmentation config list — `name`/`init_args`, nested `augmentations` for `AugmentationSequential`. |
| `mean_norm` / `std_norm` | No (from model) | Per-channel normalization override. |
| `metrics` | No | Metric names in dot notation, e.g. `["iou.macro", "f1.macro"]`. |
| `threshold_calibration_config` | No | Forwarded to `DenseCalibrationCallback`. |

**Why `class_map`/`band_map` are required, not optional counts:** a
hand-maintained `num_classes: 8` and a separately hand-maintained
`class_map` with 7 entries can silently drift out of sync — the model
builds fine, training runs, and the mismatch only shows up as a confusing
metrics/logits shape error somewhere downstream. Deriving the count from the
map removes the second, redundant source of truth entirely.

## SemanticSegmentationDataModule

`geosave_engine.ml.tasks.SemanticSegmentationDataModule` pairs with the
task above, building one `GeoDataset` per split.

| Arg | Required | Purpose |
| --- | --- | --- |
| `train_root` | For `fit` | `GeoDataset` root for the train split. |
| `val_root` | For `fit`/`validate` | `GeoDataset` root for the val split. |
| `test_root` | For `test` | `GeoDataset` root for the test split. |
| `predict_root` | For `predict` | `GeoDataset` root for the predict split. |
| `sel_bands` / `dtype_override` | No | Forwarded straight to each `GeoDataset`. |
| `pipeline` | No | A `GeoPipeline` whose `.context` supplies per-sample context — `None` omits context entirely. |
| `batch_size` / `num_workers` / `pin_memory` / `prefetch_factor` / `persistent_workers` | No | Standard `DataLoader` args. |

Each split's root is its own param, not a fixed subfolder name under one
shared root — splits routinely live in unrelated places (a `predict_root`
pointing at a fresh inference AOI has nothing to do with where
train/val/test were ingested), so baking in a naming convention would just
force awkward symlinks/copies to satisfy it. Calling `setup("fit")` without
`train_root` set raises a clear `ValueError` naming the missing arg, rather
than failing deeper inside dataset construction.

## Composition of config.yaml

The workspace splits config across three files, merged via multiple `-c`
flags — one YAML per concern, not one file with everything:

**`configs/model.yaml`** — architecture, training mechanics:

```yaml
run_name: DynamicWorld

trainer:
  max_epochs: 100
  log_every_n_steps: 10
  callbacks:                  # left blank on purpose — GeosaveCLI fills in
                               # ModelCheckpoint/LearningRateMonitor/RichProgressBar.

model:
  class_path: geosave_engine.ml.tasks.SemanticSegmentationTask
  init_args:
    stages:
      encoder: dinov3
      decoder: dpt
      head: dense
    optimizer: AdamW.split
    scheduler: CosineAnnealingLR
    loss: CELoss
    config:
      optimizer: {weight_decay: 0.01, encoder_lr: 0.0001, decoder_lr: 0.001}
      scheduler: {T_max: 100}
    threshold_calibration_config:
      metric: f1

data:
  class_path: geosave_engine.ml.tasks.SemanticSegmentationDataModule
  init_args:
    train_root: data/dynamicworld/train
    val_root: data/dynamicworld/val
    test_root: data/dynamicworld/test
    batch_size: 16
    num_workers: 4
    dtype_override:
      cloud_mask: torch.bool
      dynamicworld: torch.int64
    pipeline:
      class_path: modules.data_pipeline.Pipeline
```

**`configs/metadata.yaml`** — everything tied to *this dataset's* labels and
layer names, kept separate so it's one file to change per dataset:

```yaml
model:
  init_args:
    image_key: sentinel_2_l1c
    label_key: dynamicworld
    mask_key: cloud_mask
    ignore_index: 255
    class_map: {0: water, 1: trees, 2: grass, 3: flooded_vegetation, 4: crops, 5: shrub_and_scrub, 6: built, 7: bare}
    color_map: {0: "#419bdf", 1: "#397d49", 2: "#88b053", 3: "#7a87c6", 4: "#e49635", 5: "#dfc35a", 6: "#c4281b", 7: "#a59b8f"}
    band_map: {0: B01, 1: B02, 2: B03, 3: B04, 4: B05, 5: B06, 6: B07, 7: B08, 8: B09, 9: B10, 10: B11, 11: B12, 12: B8A}
```

Dict form, not a list — a list would make id-to-value order implicit and a
misplaced line would silently shift every id after it. A dict keeps the id
next to its value, so a mistake is visible immediately.

**`configs/augmentation.yaml`** — training-time augmentation, split out so
it can be swapped or dropped (e.g. for a `test`/`predict` run) without
touching the rest:

```yaml
model:
  init_args:
    augmentations:
      - name: RandomHorizontalFlip
        init_args: {p: 0.5}
      - name: RandomVerticalFlip
        init_args: {p: 0.5}
```

Run with all three merged — later files override earlier ones on
conflicting keys:

```bash
python main.py fit -c configs/model.yaml -c configs/metadata.yaml -c configs/augmentation.yaml
```

## LightningCLI: model args vs. data args

`main.py` (generated into every workspace, no need to edit) wires:

```python
GeosaveCLI(
    subclass_mode_model=True,
    subclass_mode_data=True,
)
```

No default `model_class`/`datamodule_class` is passed — every config
**must** give `model.class_path`/`data.class_path` explicitly (as shown
above). `model:` keys under `init_args` go straight to your
`LightningModule.__init__`; `data:` keys under `init_args` go straight to
your `LightningDataModule.__init__`.
