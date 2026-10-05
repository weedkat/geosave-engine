# Model composition

`model` owns the native model graph, input specification, and release artifacts.
Lightning training and development prediction live in `ml`; the examples using
Trainer below are consumers of model APIs. See the
[package ownership guide](../../../docs/guides/architecture.md) for the full layout.

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
inputs. A `head=True` method returns a terminal native result, validated against
its annotated outer type. One terminal result is returned unchanged; multiple
results are returned under their stage names. Intermediate named outputs remain
internal when terminals exist; encoder-only chains return named outputs.

```python
class Detector(nn.Module):
    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return self.detector(list(image))

# Heads may return different native containers in the same model call.
model = ModelChain(features=encoder, cover=segmentation, objects=detector)
outputs = model(image=image)
logits = outputs["cover"]
objects = outputs["objects"]
```

The detector in this example already decodes its outputs into tile-local boxes.
GeoSave's raw `DetectionHead` still requires its own decoding design. Terminal
tuples and dictionaries stay whole; they are not unpacked into chain features.

`Published[T]` marks constructor attributes available to later stages.
Explicit `init_args` win. Missing arguments may use one matching published
attribute; ambiguous sources and incompatible annotations raise errors.

| File | Responsibility |
| --- | --- |
| `chain/step.py` | Method declarations and runtime value checks |
| `chain/routing.py` | Method selection and dependency ordering |
| `chain/published.py` | Constructor attribute declarations and argument wiring |
| `chain/chain.py` | Registered modules and execution |
| `registry.py` | Factory registration, construction, and saved arguments |
| `release/` | Native release saving, loading, and Hub publication |

## Geospatial context

```yaml
# model_spec.yaml
inputs:
  image: !ref image
context:
  call: geosave_engine.model.encoder.prithvi.model_context
  kwargs: {row: !ref row, raster: image}
```

```python
from geosave_engine.ml.inputs import model_inputs

row = reference.loc[sample_id]
tile = row.gs.crop(parents[row.parent_id])
inputs = model_inputs(spec, tile.gs.rasters, row)
```

`inputs` binds prepared raster pixels; `context` encodes the native catalog row.
Training and inference use the same declared function. Encoder-specific
`model_context(row, *, raster="image")` functions return dictionaries of tensors.
The row's `raster_metadata` records ordered timestamps and band names per raster;
its projection fields describe the exact input grid. Neither metadata extraction
nor context encoding reads raster pixels. A row may also carry annotation columns.

| Encoder | Per-sample context |
| --- | --- |
| `PrithviTL` | `temporal_coords[T, 2]`: year and zero-based day of year; `location_coords[2]`: latitude, longitude |
| `Clay` | `time[2]`: ISO week and hour; `latlon[2]`: latitude, longitude |

Missing metadata raises an error. For Clay, select one frame before preparing the
reference. Wavelengths and GSD retain explicit encoder constructor defaults; a
CRS resolution is not assumed to be a distance in metres. Pixel normalization
remains separate from context encoding.

Explicit caching uses the same recipe and native Python arrays or numbers:

```python
cached = spec.model_context(row)
inputs = model_inputs(spec, tile.gs.rasters, row, context=cached)
```

Keep cached context associated with the same sample metadata and encoding recipe.
If a crop changes the actual input window, update its metadata before encoding
context. There is no automatic cache lookup or hidden encoder selection.

## Saving and loading native releases

```python
from geosave_engine.model.registry import build_model
from geosave_engine.model.release import load_model, load_spec, save_model

stages = {
    "encoder": {
        "name": "prithvi_tl",
        "init_args": {"pretrained": False, "num_frames": 2},
    },
}
model = build_model(stages)
save_model(model, "artifacts/model", spec="configs/model_spec.yaml")
restored = load_model("artifacts/model")
spec = load_spec("artifacts/model")
inputs = model_inputs(spec, prepared_rasters, row)
```

`build_model` constructs reproducible chains from stage specifications.
`save_model` writes `config.json`, `model.safetensors`, `model_spec.yaml`, and a
generated model card. The repository contains no executable Python. Generated
configuration records stage order, registered selectors, constructor defaults,
resolved `Published` arguments, and the exact GeoSave version needed to rebuild
the graph. Saved `pretrained` flags are false because exported weights supply the
model state.

Published stages must use registered `name` selectors implemented by the
installed GeoSave version. Workspace `class_path` stages remain valid for
experimentation, but must be promoted into the library before release. An
encoder's context functions live in its installed module, and the released
spec's `context` declares the recipe. Coordinates for a particular image are extracted
after loading, not stored in the artifact.

## Releasing a trained model

```python
from geosave_engine.ml.segmentation import supervised
from geosave_engine.model.release import publish_model

# Restore the selected checkpoint, not whichever epoch remains in memory.
task = supervised.Module.load_from_checkpoint("checkpoints/best.ckpt")
commit = publish_model(
    task.model,
    "your-account/your-model",
    spec="configs/model_spec.yaml",
)
```

Install the exact `geosave-engine[hub]` version named in the generated model
card. Pin `commit` when loading both parts of a production deployment:

```python
from geosave_engine.model.release import load_model, load_spec

spec = load_spec("your-account/your-model", revision=commit)
model = load_model("your-account/your-model", revision=commit).eval()
output = model(**model_inputs)
```

Lightning checkpoints remain training artifacts for resuming or reproducing a
run. A release contains only the native inference graph and its independently
loadable geospatial contract; it omits optimizer, scheduler, callbacks,
metrics, Trainer, and DataModule state.

### Backbone-only release

An MAE task can keep its reconstruction decoder as a separate training-only
attribute and put the deployable GFM encoder in a configured one-stage chain:

```python
# mae_task.model is build_model({"encoder": {"name": "gfm", ...}})
commit = publish_model(
    mae_task.model,
    "your-account/gfm",
    spec="configs/model_spec.yaml",
)
encoder = load_model("your-account/gfm", revision=commit).encoder
```

Publish the configured chain, not `mae_task.model.encoder` directly: the parent
chain owns the constructor recipe required to rebuild the native module.

## Predicting geospatial tiles with Lightning

```python
from lightning import Trainer
from tiler import Merger
from torch.utils.data import DataLoader, Dataset
from geosave_engine.geodata import GeoVector
from geosave_engine.ml.inputs import model_inputs

# parents holds prepared xarray stacks. layouts and padding come from native Tiler.
reference = GeoVector.from_layouts(parents, layouts, padding=padding)
reference = reference.set_index("id", drop=False)

class PredictionDataset(Dataset):
    def __len__(self):
        return len(reference)

    def __getitem__(self, position):
        row = reference.iloc[position]
        tile = row.gs.crop(parents[row.parent_id])
        return model_inputs(spec, tile.gs.rasters, row), row.id

mergers = {
    key: Merger(layout, logits=task.num_classes, window=spec.tiles.window,
                save_visits=False)
    for key, layout in layouts.items()
}
for logits, ids in Trainer().predict(task, dataloaders=DataLoader(PredictionDataset(), batch_size=8)):
    for sample_id, prediction in zip(ids, logits.detach().cpu().numpy(), strict=True):
        row = reference.loc[sample_id]
        mergers[row.parent_id].add(int(row.tile_id), prediction)
scene_logits = mergers["scene-a"].merge(extra_padding=padding["scene-a"])
```

This example assumes finite predictions and a complete prediction run. Apply
pixel transforms appropriate to the released model before its forward call.
Native Merger owns weighting and pixel placement; use the original parent's
raster grid when wrapping results. Geometry does not reconstruct layouts.

The PyTorch Dataset belongs to the consuming method. GeoSave's shared
`ml.datasets.TileDataset` has been removed. `row.gs.to_xarray()` opens both
ordinary asset rows and windowed rows. `row.gs.crop(parent)` applies an explicit
window to prepared native xarray data. Reads remain lazy until tensor
conversion. No custom tile object or automatic property-triggered loading exists.

Prediction batches are `(model_inputs, ids)` and results are `(logits, ids)`.
Supervised training/evaluation batches are `(model_inputs, target, ids, valid)`.
Validation/test blend validity-weighted logits and coverage, score only complete
parents, and retrieve original categorical labels without blending them.
Classification and detection retain IDs but require their own output policies.
Supervised scene validation/test currently require a single device. Ordinary
distributed tile sampling can split every scene across ranks, so the module
rejects that configuration instead of reporting incomplete scene metrics.
