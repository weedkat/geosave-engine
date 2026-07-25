# GeoDataset and model definition

Reference for turning ingested `.geostack` folders into training batches,
and for the standardized `SemanticSegmentationTask`/`DataModule` pair. For
the ordered how-to (pick a path, train, evaluate), see
[workflow.md](../guide/workflow.md#5-define-the-model-).

## From ingested data to tensors: GeoDataset

`geosave_engine.geodata.datasets.GeoDataset` reads a directory of
`.geostack` folders built via `GeoStack.save()` (see [pipeline.md](pipeline.md)):

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
    "anchors": dict[str, GeoAnchor],  # one bare anchor per layer, see below
}
```

| Constructor arg | Default | Purpose |
| --- | --- | --- |
| `root` | — | Directory to `rglob` for `.geostack` folders. |
| `required_layers` | `None` (all layers) | Only include anchors whose `.geostack` folder has every one of these layer names — a folder missing one is silently excluded. |
| `sel_bands` | `None` (all bands) | `{layer: [band, ...]}` — subset bands per layer. |
| `dtype_override` | `None` (saved dtype) | `{layer: torch.dtype}` — cast a layer's tensor, e.g. a saved `uint8` mask to `bool`. |
| `context_fn` | `None` (no extra keys) | `dict[LayerName, GeoTile] -> dict[str, torch.Tensor]` — merged into every sample, same function a `GeoPipeline.context` override supplies (see [pipeline.md](pipeline.md#supplying-model-specific-context)). |

**`layers` property** — the layer names present in the first sample
(`list[str]`), useful for sanity-checking a dataset without indexing into
it.

**Building your own Dataset:** subclass `GeoDataset` when you need
per-sample logic beyond what it gives you. For streaming inference over
fresh data without ingesting to disk first, there's
[ingest_to_tensor()](pipeline.md).

## What is the anchors key

Every sample always carries an `"anchors"` key — `dict[LayerName, GeoAnchor]`,
one bare anchor (no pixel data) per layer, not configurable, always present
regardless of layer content. It's a dict, not one collapsed anchor: `align()`
guarantees every layer shares the same `geobox`, but not `datetime`/
`metadata`/`polygon` — picking one representative layer would silently lose
the others' real values, so a consumer that needs exactly one picks whichever
layer's anchor it actually means (usually the model's own input layer, e.g.
`anchors[image_key]`).

`"anchors"` is not per-sample metadata extraction on its own — a model that
needs derived context (day-of-year, lat/lon as a `temporal_coords` tensor,
etc.) gets it from a `GeoPipeline.context()` override instead, merged into
the sample alongside `"anchors"` at render time. See
[pipeline.md](pipeline.md#supplying-model-specific-context) for the full
mechanism and a real example.

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
    # dict values     -> stack_samples(values)     (recurse — this is what batches "anchors")
    # anything else  -> list(values)               (gathered as-is)
```

Because `GeoDataset.__getitem__` keys samples by raw layer name, the batch
you get back has one tensor key per ingested layer, unchanged, plus
`"anchors"` recursed one level (dict values -> `stack_samples` again) into
one `list[GeoAnchor]` per layer:

```python
{
    "sentinel_2_l1c": Tensor[B, C, H, W],
    "dynamicworld": Tensor[B, 1, H, W],
    "anchors": {"sentinel_2_l1c": [GeoAnchor, ...], "dynamicworld": [GeoAnchor, ...]},  # each list length B
}
```

No new collation code needed for `"anchors"` — `stack_samples`'s existing
dict-recursion handles it for free, same as it always has for any nested
dict value.

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
| `loss` / `optimizer` / `scheduler` | No | Registry keys — see `ml/registry/{loss,optimizer,scheduler}.py` for what's registered (`CELoss`/`OHEMLoss`; `AdamW`/`Adam`/`SGD`/`RMSprop`/`Adagrad`; `CosineAnnealingLR`/`LRScheduler`). |
| `config` | No | Stage name → that stage's own constructor kwargs, plus `optimizer`/`scheduler` sub-keys for their kwargs. |
| `augmentations` | No | Kornia augmentation config list — `name`/`init_args`, nested `augmentations` for `AugmentationSequential`. |
| `mean_norm` / `std_norm` | No (from model) | Per-channel normalization override. |
| `metrics` | No | Metric names in dot notation, e.g. `["iou.macro", "f1.macro"]`. |
| `threshold_calibration_config` | No | Sweep-tuning kwargs forwarded to `ThresholdCalibrator` (`threshold_begin`/`threshold_end`/`threshold_steps`/`metric`). |
| `log_image_every_n_epochs` | No (`2`) | Epoch frequency for prediction visualization logging (`DensePredictionLogger`). |

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

`pipeline` resolves via LightningCLI's normal `class_path`/`init_args`
mechanism — same as `model.class_path`/`data.class_path` themselves, one
level deeper, no extra wiring:

```yaml
data:
  class_path: geosave_engine.ml.tasks.SemanticSegmentationDataModule
  init_args:
    train_root: data/train
    pipeline:
      class_path: modules.data_pipeline.Pipeline
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

## Writing predictions: PredictionWriter

`geosave_engine.ml.callbacks.PredictionWriter` — writes one `GeoStack` per
predicted anchor, one layer per key `predict_step` returns. Never
auto-attached (deployment-specific `output_dir` has no sensible default) —
opt in via `trainer.callbacks:`:

```yaml
trainer:
  callbacks:
    - class_path: geosave_engine.ml.callbacks.PredictionWriter
      init_args:
        output_dir: predictions
```

**Contract:** `predict_step` must return a `dict[str, Tensor]` plus one
`"anchors"` key (`list[GeoAnchor]`, one per sample) — spatial identity for
the output comes from there, not from `batch`. This callback has no
knowledge of batch keys/`image_key` at all (same reasoning as
`ThresholdCalibrator` reading `outputs['logits']`/`['label']` instead of
reaching into `batch`) — building `"anchors"` is the task's own job, since
it's the one that knows which batch layer is the real model input:

```python
output["anchors"] = batch["anchors"][self.image_key]
```

`SemanticSegmentationTask.predict_step` already does this. A custom Path B
module wiring `PredictionWriter` needs the same line in its own
`predict_step`. Missing `"anchors"` in the returned dict raises a `KeyError`
immediately, naming exactly what to add.
