# Semantic segmentation

`configs/model.yaml` configures separate Lightning model and data objects.
`configs/model_spec.yaml` accompanies the model and describes its source bands,
preprocessing and output legend. Keep the four input channels in B02, B03, B04,
B08 order and the two output classes aligned when editing these documents.

## Prepare a raster

Run the local example from this workspace; it needs no downloads or model:

```bash
uv run python scripts/prepare_example.py
```

The script creates a small lazy raster, masks nodata and unpacks reflectance.
It prints B08 values of approximately `0.6`. Use the same steps on your own raster:

```python
from geosave_engine.workflow import Processor

prepare = Processor.load("configs/model_spec.yaml", stage="preprocessing")
image = prepare({"sentinel_2_l2a": optical})["image"]
```

Inputs and results are ordinary Python objects. The returned dictionary contains
both supplied inputs and named results. Preparation stays lazy; the example
calls `.compute()` only to display pixels. Each YAML step names its result,
`call` selects a function or bound method, and `kwargs` supplies arguments.
`!ref image` refers to a prior result; ordinary text remains a literal value.

Job inputs, query/load options, output paths and Prefect settings are supplied
by the caller in Python, separately from the model document. Its source declares
the STAC collection and endpoints used for acquisition.

## Prepare training items

Write separate LitData datasets under `data/train` and `data/val` (or set the
paths in `model.yaml`). Each decoded item is `(image, target)` with an unpacked
floating-point image of shape `[4, H, W]` and an `int64` class mask of shape
`[H, W]` or `[1, H, W]`. Use pixel codes 0 for background, 1 for vegetation,
and 255 for ignored pixels. Prepare consistent tile sizes before writing.
For example, LitData can optimize your own prepared pairs:

```python
from litdata import optimize

def prepare_item(item):
    image, target = item
    return image, target

optimize(prepare_item, inputs=train_pairs, output_dir="data/train", chunk_bytes="64MB")
optimize(prepare_item, inputs=val_pairs, output_dir="data/val", chunk_bytes="64MB")
```

`modules.data.SegmentationDataModule` reads these items with LitData, collates
them into batches, then applies paired image/mask augmentation during training.
Validation uses the prepared pixels without stochastic augmentation. Batches
are `({"image": image}, target)`. Band selection, nodata handling, unpacking and
conversion to tensors belong to preparation; the training task receives tensors.
`input_size` supplies the default size for size-aware augmentations and model
construction; the data module does not implicitly resize samples.

Train from the workspace with the optional augmentation overlay:

```bash
uv run python main.py fit --config configs/model.yaml --config configs/augmentation.yaml
```

The overlay targets `data.init_args.augmentations`. The task owns its criterion,
optimizer and scheduler. Model-specific weight downloads may require access to
the selected backbone's repository.

## Interpret predictions

Processing stages run independently. A caller can extend the same document with
postprocessing calls and execute them with
`Processor.load("configs/model_spec.yaml", stage="postprocessing")` using a
supplied `prediction`. The template declares its output legend but no inference
or postprocessing calls; model loading and inference remain caller-owned.
For tiled predictions, merge logits before assigning classes. The caller owns
the accumulator and batch lifecycle; other output types need no merging steps.
