# Head-Owned Prediction Design

**Status:** draft for review, 2026-10-03

> **Tile-assembly proposal superseded, 2026-10-04.** See
> [Indexed tile reference](2026-10-04-indexed-tile-reference-design.md) and its
> [plan](../plans/2026-10-04-indexed-tile-reference.md). The `TileLayout` store,
> removal of model-spec tiling, head-owned spatial stitching, and unconditional
> `model.head` assumptions below are historical proposals, not implementation
> requirements. The described padding defect is addressed in current code;
> head interpretation and calibration ownership still need their own design.

> **Paths moved 2026-10-03** by `2026-10-03-package-restructure-design.md`. `SemanticSegmentationTask` is now `geosave_engine.ml.segmentation.supervised.Module`; `ml/lightning/tasks/` is now `ml/segmentation/supervised/`; `release` and `model_spec` are now `model.release` and `model.spec`; the `TileDataset` move described under Datasets is done.

## Purpose

A model release holds a `ModelChain` and a `model_spec.yaml`, never the Lightning
task. Anything prediction needs must live in one of those two. Today three things
have no home at deploy time:

- what the model's output means (classes or values), which only the task knows;
- calibrated class thresholds, a buffer on `SemanticSegmentationTask`;
- how tile outputs become one raster, which needs the live `Tiles` object that
  made the cut.

This design gives each a home. The head states what the output means. Tile
outputs go to a store that carries everything needed to rebuild the raster.
Stitching reads only that store.

## Scope

In scope:

- Four registered heads replacing the shared `dense` head: `segmentation`,
  `regression`, `classification`, and `detection`.
- A tile store, a writer callback, and a `stitch` function.
- A `predict` workflow flow and CLI command.
- A padding defect in `Tiles` found while testing (see Tiles).

Out of scope, each a later spec:

- Losses, metrics, tasks, and `postprocess` for `classification` and
  `detection`. Their heads are added now as modules so later work has a slot to
  build on; nothing in the repo can train or assemble their outputs yet.
- A `postprocessing` stage in the model spec.
- LitServe serving.
- Multi-process (DDP) prediction.
- Class colours on the head.

## Who owns what

| Fact | Owner | Saved in |
| --- | --- | --- |
| Input rasters, their variables, resolution, STAC recipe, preprocessing | `model_spec.yaml` | release |
| Spatial input size | encoder (`input_size`) | `config.json` |
| What the output means, class names, thresholds, output nodata | head | `config.json`, `model.safetensors` |
| Overlap, padding mode, blending window | the predict run | flow parameters |

The model spec does not mention tiling. A raster whose grid differs from the
encoder's `input_size` is cut to it.

## Data flow

```text
model_spec.load_rasters(anchor)
  -> model_spec.preprocess          whole raster, lazy
  -> Tiles(raster, encoder.input_size, overlap, mode)
  -> Trainer.predict + TileWriter   one tile store per raster, written when the raster completes
  -> head.postprocess(tiles)        stitch, decode, attach attrs; one raster at a time
  -> gs.to_cog or gs.to_zarr
```

Prediction and postprocessing are separate workflow tasks. Postprocessing needs
no GPU and can run later or elsewhere.

## Heads

`DenseHead` stays as the shared per-pixel projection but is no longer registered.
The `dense` name is removed with no alias. Four heads are registered:

| Name | Class | Consumes | Returns | `postprocess` |
| --- | --- | --- | --- | --- |
| `segmentation` | `SegmentationHead(DenseHead)` | `feature_map` | `(B, classes, H, W)` logits | yes |
| `regression` | `RegressionHead(DenseHead)` | `feature_map` | `(B, variables, H, W)` values | yes |
| `classification` | `ClassificationHead` | `pyramid` | `(B, classes)` logits | later |
| `detection` | `DetectionHead` | `pyramid` | `(B, anchors, 4 + classes)` raw predictions | later |

```python
@register_model("head", "segmentation")
class SegmentationHead(DenseHead):
    def __init__(self, classes: list[str], feature_channels: int,
                 input_size=None, hidden_channels=None, dropout=0.0,
                 nodata: int = 255): ...
    num_classes: int                      # len(classes)
    class_thresholds: Tensor              # buffer, 0.5 per class until calibrated
    def decode(self, logits: Tensor) -> Tensor: ...
    def postprocess(self, tiles: xr.Dataset, *, window: str | None) -> xr.DataArray: ...

@register_model("head", "regression")
class RegressionHead(DenseHead):
    def __init__(self, variables: list[str], feature_channels: int,
                 input_size=None, hidden_channels=None, dropout=0.0,
                 units: str | None = None): ...
    def postprocess(self, tiles: xr.Dataset, *, window: str | None) -> xr.DataArray: ...

@register_model("head", "classification")
class ClassificationHead(nn.Module):
    """Pool the deepest pyramid level and project it linearly."""
    def __init__(self, classes: list[str], pyramid_channels: list[int],
                 dropout: float = 0.0): ...

@register_model("head", "detection")
class DetectionHead(nn.Module):
    """Anchor-free YOLO-style head: one box and class branch per pyramid level."""
    def __init__(self, classes: list[str], pyramid_channels: list[int],
                 pyramid_strides: list[int], hidden_channels: int = 256): ...
```

- `ClassificationHead` and `DetectionHead` read `pyramid_channels` and
  `pyramid_strides`, which every encoder already publishes, so they chain to an
  encoder with no decoder.
- `DetectionHead` returns raw per-anchor predictions concatenated over levels.
  Box decoding, duplicate suppression, the assignment loss, and its metrics are
  not part of this spec. It is tested for output shape only.
- The ViT encoders emit every pyramid level at one stride, so `DetectionHead`
  gets no real multi-scale signal from them. It is correct but untuned there.
- `classes` is a list; a class's code is its position. A list survives the JSON
  round trip in `config.json`, which integer dict keys do not.
- `decode` is torch only: `apply_thresholds(logits, class_thresholds, nodata)`.
- `postprocess` takes one raster's tile store and returns the product raster.
  - Segmentation: `stitch` logits, `decode`, return `uint8` class codes carrying
    `Legend(class_map=...)` and `Nodata(nodata)`.
  - Regression: `stitch` values, return them with one band per name in
    `variables` and the `units` attr.
- `postprocess` mirrors the encoder's `model_context`, which already converts an
  xarray tile into model inputs. The encoder owns raster to tensors; the head
  owns tile outputs to raster.

The predict flow calls `model.head.postprocess(...)` and holds no knowledge of
head types. `ClassificationHead` and `DetectionHead` gain their own
`postprocess` when their tasks are built.

### Task changes

`SemanticSegmentationTask`:

- Drops `num_classes` and `class_thresholds`. It reads `self.model.head.num_classes`.
- Stops injecting `num_classes` into the last stage.
- Builds its metrics at the end of `configure_model`, after the model exists.
  Lightning 2.6.1 calls `setup` before `configure_model`, so metrics cannot be
  built in `setup` any more.
- Keeps `ignore_index`. It names label pixels the loss skips. The head's
  `nodata` names the value written for rejected output pixels. They are separate
  facts and are not wired together.

`ThresholdCalibrator` writes `pl_module.model.head.class_thresholds` and reads
the class count from the head. Its `num_classes` argument is removed.

```yaml
# train.yaml
model:
  class_path: geosave_engine.ml.lightning.tasks.SemanticSegmentationTask
  init_args:
    in_channels: 4
    input_size: 224
    model_chain:
      encoder: {name: dinov3}
      decoder: {name: dpt}
      head:
        name: segmentation
        init_args:
          classes: [background, oil_palm]
```

## Tiles

`Tiles` keeps its flat numbering over several rasters, `__getitem__`, `__len__`,
`locate`, and `grid`. It keeps using `tiler` for tile positions.

`overlap` takes a float fraction of the tile shape in `[0.0, 1.0)`, or an int
number of pixels. `tiler` accepts both natively, so `Tiles` passes it through.

Removed: `Tiles.merger()` and the `TileMerger` class.

Added: `Tiles.layout(ordinal) -> TileLayout`, the parameters that reproduce one
raster's cut.

```python
class TileLayout(AttrsModel):       # written at the tile store's root
    raster_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    overlap: int | float
    mode: TilingMode
    crs: str
    transform: tuple[float, float, float, float, float, float]
```

### Padding defect

`Tiles` pads a trailing tile with `xarray.pad`. For `reflect`, `symmetric`, and
`wrap`, xarray returns an array of the wrong size when the pad is larger than
the pixels available. With the default `mode="reflect"`, a 256-pixel raster cut
to 224 fails on its second tile:

```text
ValueError: conflicting sizes for dimension 'x': length 63 on 'B04' and length 224
```

`numpy.pad` handles the same case correctly. The fix pads each tile through
dask `map_blocks(np.pad, ...)` on a single-chunk tile, which stays lazy and has
numpy's semantics. Verified on a dask array: lazy, and equal to `np.pad`.

### Rasters smaller than the tile

Today `Tiles` refuses them. They are accepted and padded to the tile shape with
`mode`; stitching removes the padding. `tiler` already lays out a single tile
for such a raster and unpads it on merge.

## Tile store

Blending needs every tile's raw output, edges included, until its neighbours
exist. So tile outputs are kept as they are, in one Zarr store per raster,
stacked along a `tile` dimension. Stores sit in one directory, named by the
raster's ordinal in `Tiles`.

```text
prediction.tiles/
  0.zarr   output[tile, band, y, x]   float32
           tile[tile]                 each output's tile number
           attrs: TileLayout
  1.zarr   ...
```

- The first batch for a raster creates its store with `gs.to_zarr`. Later
  batches are added along `tile` with `io.zarr.append`.
- Tiles may arrive in any order. The `tile` coordinate records which is which.
- A store is a plain `xr.Dataset`, read back with `io.zarr.read`. `stitch` and
  `postprocess` take one.
- The store is not a map layer. The product written after `postprocess` is the
  CF/GDAL raster.

### `io.zarr.append`

GeoSave's Zarr writer writes a whole store and nothing else. Adding outputs
batch by batch needs one new function beside it:

```python
def append(raster: xr.Dataset, destination: str | PathLike[str], *, dim: str) -> Path:
    """Extend an existing Zarr store along one dimension."""
```

It wraps xarray's `to_zarr(append_dim=dim)` with the writer's own store format
and consolidation settings. The callback calls it; it does not reach past the
writer.

## Writer

`ml/callbacks/tile_writer.py`: `TileWriter(BasePredictionWriter)` with
`write_interval="batch"`.

- Reads `Tiles` from the predict dataloader's dataset.
- Groups a batch's outputs by raster with `tiles.locate(index)` and writes each
  group to that raster's store: `gs.to_zarr` with `tiles.layout(ordinal)`
  attached through `gs.rebase` the first time, `io.zarr.append` after that.
- Holds nothing between batches. Memory is bounded by one batch.
- The flow calls `Trainer.predict(..., return_predictions=False)` so Lightning
  holds nothing either.

### Predicting from a release

A release has no Lightning task, and `Trainer.predict` needs one.
`ml/lightning/tasks/prediction.py` adds `PredictionTask(model: ModelChain)`
whose only method is:

```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    model_inputs, index = batch
    return self.model(**model_inputs), index
```

`SemanticSegmentationTask.predict_step` already has this shape, so `TileWriter`
works with a training task and with a release alike.

## Stitch

```python
def stitch(tiles: xr.Dataset, *, window: StitchWindow | None = None) -> DataArray:
    """Rebuild one raster from its stored tile outputs."""
```

- Lives in `geodata/transform/tiling.py`. Takes one raster's tile store and
  nothing else.
- Reads `TileLayout` and rebuilds the same `Tiler` and a `Merger`.
- Raises when the store's `tile` numbers are not exactly the layout's, naming
  the ones missing or repeated.
- Adds every tile by its number and merges with the padding removed.
- Returns a `(band, y, x)` array on the raster's own geobox.
- Works the same on a store read from disk and on an in-memory Dataset.
- Builds its result with `array(pixels, geobox, band=...)`, the package's own
  constructor.

Verified with a throwaway script: three rasters cut by today's `Tiles`, outputs
written to Zarr with the layout parameters, then stitched in a separate process
that imports no GeoSave tiling code. Every raster rebuilt exactly, with overlap
0 and 32, with and without a `hann` window.

**Memory.** `tiler`'s `Merger` holds one raster's full output plus its weights.
A 10,980 x 10,980 scene with 10 classes needs about 5 GB. Stitching is bounded
by one raster, not by the run, and happens off the GPU.

## Datasets

`TileDataset` moves from `geodata/datasets/` to `ml/datasets/tiles.py`. It keeps
returning `(model_inputs, index)`. `geodata/datasets/` is deleted. This move is
planned separately with the STAC API change and does not wait on this spec.

## Predict flow

The flow is a thin composition of public calls. Each line below is something a
user can run in a notebook or their own script; the flow adds nothing hidden.

```python
spec   = load_spec(model_path)
model  = load_model(model_path).eval()

rasters = read_stack(stack_path).gs.rasters                 # written by `ingest`
image   = spec.preprocess(rasters)["image"]                 # the chain's raster input

tiles   = Tiles([image], model.encoder.input_size, overlap=0.25, mode="reflect")
loader  = DataLoader(TileDataset(tiles, model_context=model.encoder.model_context), batch_size=8)
writer  = TileWriter("prediction.tiles")
Trainer(callbacks=[writer]).predict(PredictionTask(model), loader, return_predictions=False)

product = model.head.postprocess(io.zarr.read("prediction.tiles/0.zarr"), window="hann")
product.gs.to_cog("prediction.tif")
```

Every step is an existing GeoSave or Lightning mechanism:

| Step | Mechanism |
| --- | --- |
| Open the release | `release.load_spec`, `release.load_model` |
| Read the ingested stack | `geodata.read_stack`, `gs.rasters` |
| Preprocess | `ModelSpec.preprocess` |
| Cut | `Tiles` |
| Tile to model inputs | `TileDataset`, `gs.to_tensor`, `encoder.model_context` |
| Run | `Trainer.predict` with a `BasePredictionWriter` callback |
| Write and read tile stores | `gs.to_zarr`, `io.zarr.append` (new), `io.zarr.read` |
| Tile layout on the store | an `AttrsModel` applied with `gs.rebase` |
| Build the product raster | `array(...)`, `gs.rebase(Legend(...))`, `Nodata` |
| Write the product | `gs.to_cog` or `gs.to_zarr` |

Smoke-tested: a store created with `gs.to_zarr` and extended batch by batch,
with tiles arriving in shuffled order, reads back through `io.zarr.read` and
stitches with a `hann` window to exactly the source raster. Also, a class raster built with `array` and
`gs.rebase(Legend)` round-trips through `gs.to_cog` and `read_raster` with its
legend, nodata, and geobox.

`workflow/flows/predict.py` wraps those lines, exposed as `geosave workflow predict`:

```python
@flow(name="predict")
def predict(stack: str, *, model: str, output: str,
            overlap: int | float = 0.25, mode: TilingMode = "reflect",
            window: StitchWindow | None = "hann",
            batch_size: int = 8) -> str: ...
```

```bash
geosave workflow ingest  --anchor ... --spec configs/model_spec.yaml --output scene.zarr
geosave workflow predict --stack scene.zarr --model artifacts/model --output prediction.tif
```

- **Input is a stack on disk, not a place and time.** `ingest` already turns an
  anchor into a stack. Predict starts from that stack, so each command does one
  thing and the downloaded data can be inspected before a GPU is involved.
- `model`: a release directory or Hub repo, holding the chain and the model spec.
- `overlap` defaults to a quarter of the tile. `window` is ignored when the
  overlap is zero.
- Task 1 preprocesses, cuts, predicts, and writes the tile store.
- Task 2 reads each raster's tile store, calls `model.head.postprocess`, and
  writes the product with `gs.to_cog`. It can be rerun against existing stores.
- The preprocessing stage must produce an output named after the chain's raster
  input, `image` for today's encoders. The flow raises, naming both sides, when
  it does not. Nothing is matched by position or guessed.

## Error handling

- A head without `postprocess` fails the flow before prediction starts, naming
  the head.
- `stitch` raises on a store with missing or repeated tiles, naming them.
- A model with no `encoder` or `head` stage, such as a monolith, fails before
  prediction starts, naming the stages it has.
- `SegmentationHead` raises when `classes` is empty or holds duplicates.
- A label raster whose `Legend` disagrees with the head's `classes` is refused
  by the task's data module. Planned with the supervised dataset, not here.

## Testing

- Heads: `decode` applies thresholds and nodata; `postprocess` returns the right
  dtype, `Legend`, and geobox; thresholds survive `save_model` and `load_model`.
  `ClassificationHead` and `DetectionHead` build from an encoder's published
  attributes and return the stated shapes.
- Task: builds from `classes`; metrics see the head's class count; the
  calibrator writes the head's buffer.
- Tiles: `reflect` on a trailing tile under half a tile; a raster smaller than
  the tile; a tile stays lazy.
- Store and stitch: write, reopen, stitch, compare, for overlap 0 and above and
  for a stack.
- Writer: a `Trainer.predict` run writes one store per raster, batch by batch.
- `io.zarr.append`: a store extended along `tile` reads back whole; appending
  to a missing store raises.
- `stitch` on a store with a tile missing raises and names it.
- Flow: ingest stack to product raster, end to end, on a small synthetic stack.

## Breaking changes

- Head name `dense` is gone. Existing configs, checkpoints, and releases naming
  it do not load.
- `SemanticSegmentationTask` no longer takes `num_classes` or `class_thresholds`.
- `ThresholdCalibrator` no longer takes `num_classes`.
- `Tiles.merger()` and `TileMerger` are gone.
- `geodata.datasets` is gone.

## Open questions

1. Default overlap of 0.25 is a common choice, not a measured one.
2. `TileDataset` hardcodes the `image` key. A model whose raster input has
   another name cannot use it yet.
3. `in_channels` is still typed by hand in `train.yaml` although the model spec
   lists the variables.
4. `RegressionHead`, `ClassificationHead`, and `DetectionHead` have no task,
   loss, or metric yet. They are covered by construction and shape tests only.
