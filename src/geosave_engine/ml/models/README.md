# Model contracts

```python
class Encoder(nn.Module):
    feature_channels: Published[int]

    @chain_step(outputs=("features",))
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

features = encoder(image)                 # Ordinary PyTorch call.
model = ModelChain(encoder=encoder, head=head)
logits = model(image=image)                # Named routing through selected steps.
```

`chain_step` attaches a `Step` declaration and returns the original method.
Direct calls keep their signatures and return values. The chain selects one
method per stage, supplies its inputs, and checks its result. A step declared
on `forward` runs through the module's normal PyTorch call hooks. Other named
methods run directly.

Without `outputs`, names are inferred from a single `return variable` or
`return first, second` statement. Supply `outputs` for expressions, branches,
or methods without readable source. Types always come from annotations;
container checks validate the outer type. Defaulted parameters are optional
inputs. A `head=True` method returns a terminal tensor.

`Published[T]` marks constructor attributes available to later stages.
Explicit `init_args` win. Missing arguments may use one matching published
attribute; ambiguous sources and incompatible annotations raise errors.

| File | Responsibility |
| --- | --- |
| `step.py` | Method declarations, input routing, and runtime value checks |
| `graph.py` | Method selection and dependency ordering |
| `published.py` | Constructor attribute declarations and argument wiring |
| `chain.py` | Registered modules, execution, and Hugging Face persistence |
| `../../registry/model.py` | Factory registration, construction, and saved arguments |

## Geospatial context

```python
samples = TileDataset(tiles, model_context=model.encoder.model_context)
loader = DataLoader(samples, batch_size=8)

for batch in loader:
    output = model(image=batch["image"], **batch["model_context"])
```

An encoder owns the conversion from a selected xarray Dataset or DataArray
into its model inputs. `TileDataset` calls the extractor while `.gs` metadata
is still available, before pixel conversion. DataLoader batches the returned
tensors. Training dataset adapters retain this mapping under
`batch["model_context"]`; the Lightning task forwards it to the model.

| Encoder | Per-sample context |
| --- | --- |
| `PrithviTL` | `temporal_coords[T, 2]`: year and zero-based day-of-year; `location_coords[2]`: latitude, longitude |
| `Clay` | `time[2]`: ISO week and hour; `latlon[2]`: latitude, longitude |

These extractors encode coordinate labels, including their original frame
order, and use the tile's geographic centroid. Missing or invalid metadata
raises an error. For Clay, select one frame with `isel(time=...)` before
tiling, so its image tensor has no time axis. For a DataTree/GeoStack, supply a
callback that explicitly selects the relevant group.

Prithvi accepts geodata's batched temporal layout `(B, T, C, H, W)` and moves
the time axis for its underlying backbone. Single-frame `(B, C, H, W)` inputs
remain supported. Clay's wavelengths and GSD remain explicit constructor
values; CRS resolution is not assumed to be a distance in metres. Image
normalization is separate and remains caller-owned.

## Saving, uploading, and loading

```python
from geosave_engine.ml.models.contract import ModelChain
from geosave_engine.ml.registry import StageSpec

stages: dict[str, StageSpec] = {
    "encoder": {
        "name": "prithvi_tl",
        "init_args": {"pretrained": False, "num_frames": 2},
    },
}
model = ModelChain(stages=stages)
model.save_pretrained("artifacts/model")
restored = ModelChain.from_pretrained("artifacts/model", local_files_only=True)

# Run these when publishing to your own repository:
model.push_to_hub("your-account/your-model")
restored = ModelChain.from_pretrained("your-account/your-model", revision="main")
samples = TileDataset(tiles, model_context=restored.encoder.model_context)
```

Construct publishable chains from stage specifications. Their artifact contains
`config.json`, `model.safetensors`, and a generated model card. Configuration
records stage order, selectors, constructor defaults, and resolved Published
arguments. The saved `pretrained` flags are false: the exported weights supply
the model state. Loading is strict by default.

The receiving environment needs GeoSave Engine and the stage implementations.
Built-in named factories are registered by the package. Custom named factories
must be registered by their package; custom `class_path` classes must be
importable. The mixin does not bundle Python source or pickle extractor functions.
An encoder's `model_context` method returns with its class. Coordinates for a
particular image are extracted after loading, not stored in the model artifact.

## Releasing a trained model

```python
from geosave_engine.ml.tasks.semantic_segmentation import SemanticSegmentationTask
from geosave_engine.ml.huggingface import GeoSaveModel

# Restore the selected checkpoint, not whichever epoch remains in memory.
task = SemanticSegmentationTask.load_from_checkpoint("checkpoints/best.ckpt")
published = GeoSaveModel.from_chain(task.model)
published.save_pretrained("artifacts/release")
published.push_to_hub("your-account/your-model")
```

Install `geosave-engine[hub]` for the optional Transformers integration.
`from_chain` shares the existing modules and weights; it does not rebuild or
reinitialize them. Configuration stores stages as a list so JSON key sorting
cannot change construction order. The exported source uses installed GeoSave
and stage packages; it does not bundle those dependencies.

```python
from transformers import AutoModel
from geosave_engine.ml.huggingface import GeoSaveModel  # Registers installed classes.

model = AutoModel.from_pretrained("your-account/your-model", trust_remote_code=False)
output = model(image=image, **model_context)
```

Alternatively, loading without importing the integration uses the exported
adapter code: `AutoModel.from_pretrained(repo_id, revision=commit_sha,
trust_remote_code=True)`. Both forms require GeoSave and custom stage packages.
The original mixin format still loads through `ModelChain.from_pretrained`;
it is a separate format from the Transformers export.

A Lightning `.ckpt` records training state and is appropriate for resuming or
reproducing training. It may be shared for that purpose, but the inference
release contains `config.json`, safetensors weights, adapter source, and a
model card. `push_to_hub` exports the model into a temporary directory; it does
not upload a training checkpoint directory. Checkpoint creation stays under
Lightning's `ModelCheckpoint`; publication is an explicit operation after
selecting a checkpoint.

Task-owned thresholds, labels, and preprocessing are not included by exporting
`task.model`. Document required bands, units, scaling, normalization, class
labels, model context, and any decision thresholds in the release's model card.
The adapter forwards model inputs unchanged and does not claim Transformers
pipeline or Transformers Trainer support.

See Hugging Face's [custom model publishing guide](https://huggingface.co/docs/transformers/en/custom_models).

## Predicting geospatial tiles with Lightning

```python
from lightning import Trainer
from torch.utils.data import DataLoader
from geosave_engine.geodata.datasets import TileDataset
from geosave_engine.geodata.transform.tiling import Tiles

# task is a SemanticSegmentationTask restored from the selected checkpoint.
tiles = Tiles([scene], task.input_size, overlap=32)
samples = TileDataset(tiles, model_context=task.model.encoder.model_context)
loader = DataLoader(samples, batch_size=8)
merger = tiles.merger(window="hann")

for batch in Trainer().predict(task, dataloaders=loader):
    merger.add(dict(zip(
        batch["index"].tolist(), batch["logits"].cpu().numpy(), strict=True,
    )))
scene_logits = merger.merge()
```

Select the model's bands and numerical representation before tiling. Supply
`model_context` only for encoders that need it; extraction runs on each tile's
xarray object. A stack needs a callback that selects its input group.
`Trainer.predict` collects returned batches in memory; the direct model loop
above can feed the merger incrementally for larger runs.

Prediction batches use TileDataset's `image`, `index`, and optional
`model_context`. Training, validation, and test batches retain the supervised
`layers` mapping. Validation and test evaluate those prepared tiles directly;
metrics count overlapping pixels more than once if the dataset repeats them.
For full-scene metrics, stitch logits first and evaluate against scene labels.

Apply `task.postprocess` only after stitching, supplying the scene nodata mask
if needed. Class assignment is nonlinear: neither class IDs nor thresholded
labels should be averaged across overlapping tiles. For one reconstructed
scene, the logits tensor supplied to `postprocess` has shape `(1, classes, y, x)`;
its threshold buffer and mask must be on the same device. Geodata owns the
result's coordinates and georeferenced writing.

`forward_sliding`, standalone `task.predict`, `overlap_ratio`,
`sliding_batch_size`, and `mask_key` have been removed. Configure overlap on
`Tiles`, batch size on `DataLoader`, and masking when processing the reconstructed
scene. Older task configurations/checkpoints with these constructor arguments
need them removed before loading.
