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
from geosave_engine.model_spec import ModelSpec

spec = ModelSpec.load("configs/model_spec.yaml")
image = spec.preprocess({"sentinel_2_l2a": optical})["image"]
```

Job areas, dates, query/load options, destinations, and Prefect settings are
caller-owned flow parameters. Collection identity, endpoint priority, source
requirements, and model-specific preparation stay with the model document.

## Prepare training items

The configured `SemanticSegmentationDataModule` defines the Lightning lifecycle
and ordinary PyTorch `DataLoader` behavior. Its storage-specific `_dataset`
method is intentionally a skeleton. Subclass it to construct a
`torch.utils.data.Dataset` for each path, then point `data.class_path` at that
subclass.

Each dataset item is `({"image": image, ...}, target)`, where named inputs match
the model chain. For this example, `image` is a floating-point tensor with shape
`[4, H, W]`, and `target` is an `int64` class mask with shape `[H, W]` or
`[1, H, W]`. Use pixel code 255 for ignored pixels.

Train after selecting the data-module subclass:

```bash
uv run python main.py fit --config configs/train.yaml
```

## Release and inference

Restore the selected checkpoint, then publish its native inference graph with
the model-owned processing contract:

```python
from geosave_engine.ml.lightning.tasks import SemanticSegmentationTask
from geosave_engine.release import publish_model

task = SemanticSegmentationTask.load_from_checkpoint("checkpoints/best.ckpt")
commit = publish_model(
    task.model,
    "your-account/your-model",
    spec="configs/model_spec.yaml",
)
```

Deployment loads the two release parts independently at the returned commit:

```python
from geosave_engine.release import load_model, load_spec

spec = load_spec("your-account/your-model", revision=commit)
model = load_model("your-account/your-model", revision=commit).eval()
```

The spec owns raster requirements, acquisition, and preprocessing. The native
model accepts named tensor inputs and returns raw logits. Task-specific
inference code owns tiling, iterative execution, aggregation, and output
interpretation when those behaviors are needed.
