# StackDataset and model definition

Reference for turning ingested `.zarr` stores into training batches,
and for the standardized `SemanticSegmentationTask`/`DataModule` pair. For
the ordered how-to (pick a path, train, evaluate), see
[workflow.md](../guide/workflow.md#5-define-the-model-).

Everything on this page is `geodata.datasets` — training-only, whole-sample
reads. See [geotile.md](geotile.md#small-vs-big-datasets-vs-datastore) for
how that differs from `geodata.datastore` (windowed, serving/viewing).

> **Status:** `geodata.datasets` is mid-rehaul — `StackDataset` (below) and
> `StoreDataset` (a `SampleStore`-backed counterpart, for bulk litdata
> storage instead of one `.zarr` per anchor) are both currently skeletons
> (`NotImplementedError` bodies). The shape described here is the settled
> target design, not yet runnable. `GeoDataset`/`NonGeoDataset`/
> `IntersectionDataset`/`BaseDataset`/`TableDataset`/`YoloDataset`/
> `CocoDataset` — an earlier generation of this module — are gone: `SampleStore`
> is now the primary bulk-storage mechanism, and joining separate
> single-modality datasets by a shared key was judged out of scope for this
> library (a data-team concern, not a training-time one).

## From ingested data to tensors: StackDataset

`geosave_engine.geodata.datasets.StackDataset` reads a directory of
`.zarr` stores built via `GeoStack.to_zarr()` (see [pipeline.md](pipeline.md)):

```python
from geosave_engine.geodata.datasets import StackDataset

ds = StackDataset("data/train")
```

Discovery is `root.rglob("*.zarr")` — every anchor store, at any
nesting depth (see
[geotile.md](geotile.md#the-zarr-store-convention) for why the
`.zarr` suffix exists and how nested grouping works). `__getitem__`
delegates straight to `GeoStack.to_tensor`:

```python
{
    "sentinel_2_l1c": torch.Tensor,             # [C, H, W]
    "dynamicworld": torch.Tensor,               # [1, H, W]
    "geobox": dict,                             # this stack's one shared geobox, JSON-safe
    "geotags": dict[str, dict],                 # per-layer geotag, JSON-safe, see below
    # plus whatever keys the ingesting GeoStack's own .context carried, if any
}
```

| Constructor arg | Default | Purpose |
| --- | --- | --- |
| `root` | — | Directory to `rglob` for `.zarr` stores. |
| `required_layers` | `None` (all layers) | Only include anchors whose `.zarr` store has every one of these layer names (as Zarr groups) — a store missing one is silently excluded. |
| `sel_bands` | `None` (all bands) | `{layer: [band, ...]}` — subset bands per layer. |
| `dtype_override` | `None` (saved dtype) | `{layer: torch.dtype}` — cast a layer's tensor, e.g. a saved `uint8` mask to `bool`. |

No `context_fn` here — a `GeoStack`'s `context` (see
[the geobox/geotags section below](#what-are-the-geoboxgeotags-keys)) is
computed once at ingest time and persisted in the `.zarr` itself; `render`
just reads it back, nothing to configure per dataset.

**Building your own Dataset:** subclass `StackDataset` when you need
per-sample logic beyond what it gives you. For streaming inference over
fresh data without ingesting to disk first, there's
[ingest_to_tensor()](pipeline.md).

## What are the geobox/geotags keys

Every sample always carries `"geobox"` and `"geotags"` — plain, JSON-safe
dicts (not configurable, always present regardless of layer content):
`"geobox"` is this stack's one shared geobox (`{"shape", "affine", "crs", "centroid"}`
— `"centroid"` is a precomputed WGS84 `(lon, lat)`, done once here since
reprojecting a GeoBox is real work, not something a model's hot forward
loop should redo per call);
`"geotags"` is `dict[LayerName, dict]`, one geotag per layer. Geobox is
genuinely shared — `align()` guarantees every layer sits on the same grid —
but datetime/metadata/polygon aren't, so geotags stays per-layer rather than
one collapsed value: picking one representative layer would silently lose
the others' real values, so a consumer that needs exactly one picks
whichever layer's geotag it actually means (usually the model's own input
layer, e.g. `geotags[image_key]`).

Neither key is per-sample metadata extraction on its own — a model that
needs derived context (day-of-year, lat/lon as a `temporal_coords` tensor,
etc.) gets it from a `GeoPipeline.context()` override instead. Unlike
`"geobox"`/`"geotags"`, this isn't computed at render time: `GeoStack` has
its own `context: dict[str, torch.Tensor]` field, set once when
`GeoPipeline.ingest()` builds the stack, persisted through `to_zarr`/
restored by `from_zarr` (a root-level attr), and merged into
`to_tensor()`/`to_numpy()`'s output every time after that with no
recomputation — reading a saved `.zarr` a thousand times over a training
run costs the same as reading it once. See
[pipeline.md](pipeline.md#supplying-model-specific-context) for the full
mechanism and a real example.

## The zarr format

Ingestion writes zarr, not GeoTIFF, because a layer can carry a time
dimension (`(time, band, y, x)`) natively — a plain GeoTIFF can't.
`GeoTile.to_zarr`/`from_zarr` (see [geotile.md](geotile.md)) round-trip the
array plus datetime, metadata, and polygon footprint as store attributes —
no guessing from the path.

`StackDataset`'s discovery pass opens every store with `load_data=False` —
header-only, geobox and datetime read from attrs without touching pixels.
Scanning a large `data/train/` directory to build the sample index is cheap
for exactly this reason. Pixels are only materialized when `__getitem__`
calls `tile.to_tensor()`, which clips to the tile's bbox and reads — so the
cost of one epoch is proportional to what the `DataLoader` actually visits,
not to how many `.zarr` stores exist on disk.

## Bulk litdata storage: StoreDataset

For data that doesn't need per-anchor `.zarr` folders — high-throughput
training reads, or an existing `SampleStore` bulk export — `StoreDataset`
wraps a `litdata.StreamingDataset` over a `SampleStore`. Complete-sample-only,
same as `StackDataset`: neither class joins separate single-modality
datasets by a shared key (see the status note above) — a sample carries
everything it needs, or it belongs in a different `SampleStore`/`.zarr`
anchor.

## stack_samples

The `DataLoader`'s `collate_fn`. Turns a list of per-sample dicts into one
batched dict, recursively, by value type:

```python
def stack_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    # tensor values  -> torch.stack(values)      (one stacked Tensor)
    # dict values     -> stack_samples(values)     (recurse — this is what batches "geotags")
    # anything else  -> list(values)               (gathered as-is)
```

Because `StackDataset.__getitem__` keys samples by raw layer name, the batch
you get back has one tensor key per ingested layer, unchanged, plus
`"geobox"` gathered as a `list[dict]` (length B, one per sample — not a
shared value, since different samples in a batch cover different places)
and `"geotags"` recursed one level (dict values -> `stack_samples` again)
into one `list[dict]` per layer:

```python
{
    "sentinel_2_l1c": Tensor[B, C, H, W],
    "dynamicworld": Tensor[B, 1, H, W],
    "geobox": [dict, ...],  # length B
    "geotags": {"sentinel_2_l1c": [dict, ...], "dynamicworld": [dict, ...]},  # each list length B
}
```

No new collation code needed for either key — `stack_samples`'s existing
dict-recursion and plain-value gathering handle them for free, same as it
always has for any nested dict value.

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
| `image_key` / `label_key` / `mask_key` | No (`image`/`label`/`mask`) | Batch keys — point these at your `StackDataset`'s raw layer names instead of requiring a renaming step. |
| `ignore_index` | No (`255`) | Class index excluded from loss and metrics. |
| `input_size` | No (`224`) | Spatial patch size for sliding-window inference. |
| `loss` / `optimizer` / `scheduler` | No | Registry keys — see `ml/registry/{loss,optimizer,scheduler}.py` for what's registered (`CELoss`/`OHEMLoss`; `AdamW`/`Adam`/`SGD`/`RMSprop`/`Adagrad`; `CosineAnnealingLR`/`LRScheduler`). |
| `config` | No | Stage name → that stage's own constructor kwargs, plus `optimizer`/`scheduler` sub-keys for their kwargs. |
| `augmentations` | No | Kornia augmentation config list — `name`/`init_args`, nested `augmentations` for `AugmentationSequential`. |
| `mean_norm` / `std_norm` | No (from model) | Per-channel normalization override. |
| `metrics` | No | Metric names in dot notation, e.g. `["iou.macro", "f1.macro"]`. |
| `threshold_calibration_config` | No | Sweep-tuning kwargs forwarded to `ThresholdCalibrator` (`threshold_begin`/`threshold_end`/`threshold_steps`/`metric`). |
| `class_thresholds` | No (`0.5` per class) | Initial per-class confidence threshold, one per `class_map` entry — a placeholder `ThresholdCalibrator` overwrites, or a real value if you already know good thresholds. |
| `log_image_every_n_epochs` | No (`2`) | Epoch frequency for prediction visualization logging (`DensePredictionLogger`). |

**Why `class_map`/`band_map` are required, not optional counts:** a
hand-maintained `num_classes: 8` and a separately hand-maintained
`class_map` with 7 entries can silently drift out of sync — the model
builds fine, training runs, and the mismatch only shows up as a confusing
metrics/logits shape error somewhere downstream. Deriving the count from the
map removes the second, redundant source of truth entirely.

## SemanticSegmentationDataModule

`geosave_engine.ml.tasks.SemanticSegmentationDataModule` pairs with the
task above, building one `StackDataset` per split.

| Arg | Required | Purpose |
| --- | --- | --- |
| `train_root` | For `fit` | `StackDataset` root for the train split. |
| `val_root` | For `fit`/`validate` | `StackDataset` root for the val split. |
| `test_root` | For `test` | `StackDataset` root for the test split. |
| `predict_root` | For `predict` | `StackDataset` root for the predict split. |
| `sel_bands` / `dtype_override` | No | Forwarded straight to each `StackDataset`. |
| `batch_size` / `num_workers` / `pin_memory` / `prefetch_factor` / `persistent_workers` | No | Standard `DataLoader` args. |

Each split's root is its own param, not a fixed subfolder name under one
shared root — splits routinely live in unrelated places (a `predict_root`
pointing at a fresh inference AOI has nothing to do with where
train/val/test were ingested), so baking in a naming convention would just
force awkward symlinks/copies to satisfy it. Calling `setup("fit")` without
`train_root` set raises a clear `ValueError` naming the missing arg, rather
than failing deeper inside dataset construction.

No `pipeline` arg here — a model-context-needing `Pipeline` is configured
once, at ingest time (see [pipeline.md](pipeline.md#supplying-model-specific-context)),
not re-wired into every training run. `StackDataset` just reads whatever
`GeoStack.context` each `.zarr` already carries:

```yaml
data:
  class_path: geosave_engine.ml.tasks.SemanticSegmentationDataModule
  init_args:
    train_root: data/train
```

## Model construction: registry, model_context, ContextChain

What `stages: {encoder: dinov3, decoder: dpt, head: dense}` actually builds
— `geosave_engine.ml.models.contract`/`geosave_engine.ml.registry`. Path A
(`SemanticSegmentationTask`) and Path B (your own `LightningModule`) both go
through this; Path B just calls `build_model` directly instead of a task
doing it for you.

**Registry** (`ml/registry/model.py`) — `@register_model(stage, name)`
registers an `nn.Module` class under `MODEL_REGISTRY[stage][name.upper()]`.
`build_model(stages, config)` resolves each stage's class (registry key or
the class itself), auto-wires `{stage}_{attr}` params from already-built
earlier stages (e.g. a decoder's `encoder_out_channels` pulled straight off
the built encoder instance, no hand-typed duplicate), merges in
`config[stage]`, and constructs it — the first stage gets `is_entry = True`.
Currently registered:

| Stage | Keys |
| --- | --- |
| `encoder` | `prithvi`, `prithvi_tl`, `dinov3`, `clay` |
| `decoder` | `dpt`, `unet` |
| `head` | `dense` |
| `monolith` (single-stage chain) | `granite_geospatial_biomass` (IBM's Prithvi-based biomass model) |

**`@model_context`** (`ml/models/contract/context.py`) — marks one method
per chain stage as its real forward step. `requires` comes from the
method's own parameter names/types (its real signature, not a hand-typed
dict); `provides` comes from its `-> tuple[T1, T2, ...]` return annotation
zipped against its own `return name1, name2, ...` statement (via `ast`, not
executed) — one source of truth for both directions, nothing to keep in
sync by hand. Strict on purpose: exactly one `return` statement, a
fixed-arity `tuple[...]` of concrete types, arity matching the return
statement — anything looser is a `TypeError` at decoration time, not a
silently-accepted edge case.

```python
@model_context()
def forward_pyramid(self, image: torch.Tensor) -> tuple[list, list]:
    ...
    return pyramid, prefix_tokens
```

`head=True` marks a terminal method — returns a raw `torch.Tensor`, ends
the chain, instead of a dict merged into the shared context.

**`ContextChain`** (`ml/models/contract/chain.py`) — the `nn.Module`
`build_model` hands back. Builds a bipartite key/method graph from every
submodule's `@model_context` method(s), walks it by topological generation
to pick one method per module and a valid call order, then `forward` runs
each in order, merging `{**ctx, **result}` after each step (immutable —
prior keys survive, branching/merging both fall out of the same graph walk,
no special-casing). A module offering several candidate methods (alternate
accepted input shapes) is fine — whichever one's `requires` the graph can
satisfy first gets picked; two candidates surfacing in the same generation
is a genuine ambiguity, raised as `TypeError`, not guessed at. Verified
`torch.compile`/`torch.export`/ONNX-export compatible — Dynamo-based tracing
tolerates the dynamic dispatch inside `ContextChain.forward` the way
`torch.jit.script`'s static AST compiler wouldn't.

**Clay's sensor-agnostic constructor** — `encoder/clay.py`'s `Clay` takes
`in_channels`/`waves`/`gsd` as plain caller-supplied numbers, no `modality`
string, no sensor-catalog lookup inside `ml/` at all (mirrors `Prithvi`/
`DINOv3` taking a plain `in_channels`). Resolve those from a real sensor via
`geosave_engine.geodata.sensors` — `sensor_bands`, `sensor_gsd`,
`band_wavelengths`, `band_mean`, `band_std`, `band_gsd` — in your
`config[stage]`, not inside the model class:

```yaml
config:
  encoder:
    in_channels: 10
    waves: [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]  # geosave_engine.geodata.sensors.band_wavelengths("sentinel-2-l2a", bands)
    gsd: 10.0  # geosave_engine.geodata.sensors.sensor_gsd("sentinel-2-l2a")
```

Clay doesn't implement `Normalization` (no `img_mean`/`img_std` attribute)
— set `mean_norm`/`std_norm` explicitly on `SemanticSegmentationTask`
instead (same `geosave_engine.geodata.sensors.band_mean`/`band_std` source).

## Composition of config.yaml

The workspace splits config across three files, merged via multiple `-c`
flags — one YAML per concern, not one file with everything:

**`configs/model.yaml`** — architecture, training mechanics:

```yaml
model_name: DynamicWorld

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
