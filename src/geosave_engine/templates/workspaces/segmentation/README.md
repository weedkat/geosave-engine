# Semantic segmentation

`configs/train.yaml` configures the Lightning Trainer, model, and data objects.
`configs/model_spec.yaml` accompanies the model and declares its required source
bands, STAC acquisition recipe, and lazy preprocessing.

## Prepare a raster

Run the local example from this workspace; it needs no downloads or model:

```bash
uv run python scripts/prepare_example.py
```

Input is a packed, lazy four-band Sentinel-2 `xarray.Dataset`. Output is a
sample-ready lazy Dataset with nodata masked and reflectance unpacked. The
example computes only the small result it prints.

```python
from geosave_engine.model.spec import ModelSpec

spec = ModelSpec.load("configs/model_spec.yaml")
image = spec.preprocess({"sentinel_2_l2a": optical})["image"]
```

Job areas, dates, destinations, and Prefect settings are caller-owned flow
parameters. Collection identity, endpoint priority, query/load options, source
requirements, and model-specific preparation stay with the model document.

## Prepare training items

`prepare_dense_data` writes one folder per sample, holding one raster per layer,
and a `manifest.parquet` listing the samples as STAC items. Run it once per
split, so training and validation each have their own manifest:

```bash
uv run geosave workflow prepare-dense-data --labels labels/train --output data/train --spec configs/model_spec.yaml
uv run geosave workflow prepare-dense-data --labels labels/val --output data/val --spec configs/model_spec.yaml
```

Samples stay raw on disk. `supervised.DataModule` reads them through
`configs/model_spec.yaml`:

| Section | What it does | Where it runs |
| --- | --- | --- |
| `preprocessing` | nodata, unpacking, band order | per sample, reading from disk |
| `tiles` | cuts every sample into model-sized tiles | reading from disk |
| `inputs` | names what the model chain takes | per tile |
| `transforms` | tensor transforms such as normalization | on the device, every split |

`data.init_args.augmentations` in `configs/train.yaml` run on the device while
training only, on the image and the target together, before `transforms`.

A batch is `(model_inputs, target, ids, valid)`. `model_inputs` holds the inputs the
spec lists; `target` is an `int64` class mask shaped `[batch, H, W]`, with pixel
code 255 ignored. Validation merges tile predictions back onto each sample and
scores every sample whole.

To change the encoder, edit `model.init_args.model_chain.encoder` in
`configs/train.yaml`. Enable the matching row-based `context` recipe in the spec
for Prithvi TL or Clay. The default DINOv3 starter leaves context disabled.
Clay wavelengths and GSD are explicit encoder constructor settings.

```bash
uv run python main.py fit --config configs/train.yaml
```

## Release and inference

Restore the selected checkpoint, then publish its native inference graph with
the model-owned processing contract:

```python
from geosave_engine.ml.segmentation import supervised
from geosave_engine.model.release import publish_model

task = supervised.Module.load_from_checkpoint("checkpoints/best.ckpt")
commit = publish_model(
    task.model,
    "your-account/your-model",
    spec="configs/model_spec.yaml",
)
```

Deployment loads the two release parts independently at the returned commit:

```python
from geosave_engine.model.release import load_model, load_spec

spec = load_spec("your-account/your-model", revision=commit)
model = load_model("your-account/your-model", revision=commit).eval()
```

The spec owns raster requirements, acquisition, and preprocessing. The native
model accepts named tensor inputs and returns raw logits. Task-specific
inference code owns tiling, iterative execution, aggregation, and output
interpretation when those behaviors are needed.

Library supervised batches contain `(model_inputs, target, ids, valid)`.
The reference ID identifies the prepared scene/frame tile; validation/test
assemble logits and score the original parent labels. Validity excludes input
nodata before zero filling. Augmentation stays in the training DataModule.

For development prediction, build a method-owned PyTorch Dataset that reads
`reference.loc[id].gs.crop(parent)` and prepares tensors with
`geosave_engine.ml.inputs.model_inputs(spec, tile.gs.rasters, row)`. Ordinary
asset rows and tiled rows use the same raster reader; tiled rows add a pixel
window. The shared `ml.datasets.TileDataset` has been removed.

Use `spec.chips.tiler(shape)` to construct native spatial Tilers.
`transform.chip.chip_windows(parents, tilers, padding=...)` creates the reference
without reading pixels. Each ID identifies a parent, native tile ID, window,
exact grid, and per-raster timestamps/bands. Supervised references retain
annotations, `source_id`, and `source_assets` for provenance. Those source paths
do not store the prepared pixels: use `row.gs.crop(parents[row.parent_id])`.
Register saved prepared parents separately to obtain readable `assets`.

Enable the encoder's row-based `context` recipe in `model_spec.yaml` when its
metadata is available. The template leaves context disabled for its default
DINOv3 encoder. Context encoding is shared by training and inference; cached
values are an optional explicit argument, not a different inference recipe.

Native `tiler.Merger` accepts the row's integer tile ID and numpy prediction.
Convert tensors with `detach().cpu().numpy()` in ML. Keep tilers and halo widths
with the job and pass padding to `Merger.merge(extra_padding=...)`.

Supervised validation/test own completion and validity policy. They accumulate
logits and coverage using the same native taper and score original parent
labels only after every tile arrives.
Validation/test require a single device until scene partitioning across ranks
is supported. Prediction still returns indexed logits for caller-owned merging.

`ImageAugmenter` remains available from `geosave_engine.ml.transforms` for
YAML-configured native Kornia augmentation, including nested pipelines. The
DataModule owns joint image/target and temporal handling, passing `data_keys`
at the augmentation call. Multiple inputs require explicit keys; a single
image can omit them. The shared augmenter
also supports Pascal VOC `bbox_xyxy`, COCO `bbox_xywh`, and normalized
`bbox_yolo` boxes for custom training methods; YOLO class IDs use a separate
`class` or `label` tensor.
