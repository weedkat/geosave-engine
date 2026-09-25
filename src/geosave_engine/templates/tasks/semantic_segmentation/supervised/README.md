# Semantic segmentation

`configs/model.yaml` configures the Lightning model and data objects.
`configs/model_spec.yaml` accompanies the model and declares its required source
bands, lazy preprocessing, and future per-sample inference input.

## Prepare a raster

Run the local example from this workspace; it needs no downloads or model:

```bash
uv run python scripts/prepare_example.py
```

Input is a packed, lazy four-band Sentinel-2 `xarray.Dataset`. Output is a
sample-ready lazy Dataset with nodata masked and reflectance unpacked. The
example computes only the small result it prints.

```python
from geosave_engine.workflow.specs import ModelSpec
from geosave_engine.workflow.tasks.preprocess import Preprocessor

spec = ModelSpec.load("configs/model_spec.yaml")
image = Preprocessor(spec).run({"sentinel_2_l2a": optical})["image"]
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
uv run python main.py fit --config configs/model.yaml
```

## Inference and postprocessing status

The model spec declares `image.gs.to_tensor(dtype="float32")` as the future
inference-input conversion. The current workflow validates and round-trips that
call but does not execute inference: bounded sampling, batching, model loading,
device policy, and prediction deployment still need their own design.

`postprocessing` is intentionally empty. Prediction interpretation, tiling, and
merging are not implemented by this workflow revision.
