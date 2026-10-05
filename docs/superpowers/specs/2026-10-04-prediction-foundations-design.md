# Prediction foundations

Status: native terminal results and the indexed reference implemented,
2026-10-04. Keyed datasets and reference-driven dense assembly migrated on
2026-10-05; workflow signatures remain proposals. Implementation order and gates are in the
[plan](../plans/2026-10-04-prediction-foundations.md).

## Caller code

```python
# Terminal outputs may be native tensors or native detection containers.
model = ModelChain(
    features=Encoder(),
    cover=SegmentationHead(...),
    amount=RegressionHead(...),
    objects=Detector(),
)
outputs = model(image=image)
logits = outputs["cover"]       # Tensor[B, classes, H, W]
values = outputs["amount"]      # Tensor[B, variables, H, W]
objects = outputs["objects"]    # list[dict[str, Tensor]] in this detector

# Concrete first reference addition; the dataset migration follows later.
tiles = spec.tiles.cut([scene_a, scene_b])
reference = tiles.reference(parent_ids=["scene-a", "scene-b"])
lookup = reference.set_index("id", drop=False, verify_integrity=True)
rows = lookup.loc[tile_ids]     # IDs travel beside model inputs and outputs.
```

`Detector` here means a native detector whose output is already decoded into
tile-local boxes. GeoSave's current raw `DetectionHead` does not implement that
contract. Its decoding and training loss require a separate design.

The later keyed dataset constructs its reference from the same cut and parent
IDs, so changing the order of an unrelated table cannot silently mispair pixels:

```python
# Implemented keyed dataset API.
dataset = TileDataset(tiles, spec=spec, parent_ids=["scene-a", "scene-b"])
reference = dataset.reference
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for inputs, ids in loader:
    outputs = model(**inputs)
    rows = reference.set_index("id").loc[list(ids)]
    # Explicit consumers receive the selected output and its matching rows.
```

Keep `reference` as a native GeoDataFrame with an `id` column. Persistence uses
`index=False`; runtime lookup uses `set_index("id", verify_integrity=True)`.
DataLoader positions are local integers; returned IDs identify reference rows.
The dataset-generated reference must be treated as immutable while it is read.

## Maintainer code: model results

```python
class Detector(nn.Module):
    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return self.detector(list(image))

# ModelChain.forward: decide using the declaration, not the value's type.
for step in self._steps:
    result = step.method.invoke(step.module, context)
    if step.method.head:
        heads[step.stage] = result
    else:
        context.update(result)
        produced.update(result)

if len(heads) == 1:
    return next(iter(heads.values()))
return heads if heads else produced
```

The existing `head=True` flag declares a terminal result, not a raster kind or
a tensor-only restriction. Preserve the annotated return type on `Step` and
validate its outer runtime type through the existing validator. A terminal
tuple is one native result, not several named intermediate outputs. Missing,
`None`, and unsupported runtime annotations are rejected at declaration.
Container elements remain the native module's responsibility, matching current
outer-type validation; do not add recursive schema machinery.

One terminal result is returned unchanged, including a detection list or dict.
Several terminal results are returned under their stage names. Encoder-only
chains continue returning named produced values. Named intermediate outputs
are intentionally excluded when terminal results exist. Consequently a detector
must explicitly declare itself terminal to participate in a mixed result;
automatically returning all intermediate features would change a different
contract. No new model-result wrapper or output registry is necessary.

`chain_step(head=True, outputs=(...))` stays invalid. Routing and constructor
publication keep their existing behavior. Chain diagnostics show the terminal's
actual annotation rather than always saying `Tensor`.

## Maintainer code: raster assembly and scoring

The implemented merger replaces live-cut assembly. Tiler still generates
windows for lazy cutting; assembly consumes saved reference rows independently.

```python
class TileMerger:
    def __init__(
        self,
        reference: gpd.GeoDataFrame,
        parents: Mapping[str, xr.Dataset | xr.DataArray],
        *,
        window: StitchWindow | None = None,
        overlap: int = 0,
        leading_dims: tuple[str, ...] | None = None,
    ) -> None: ...

    def add(
        self,
        results: Mapping[str, np.ndarray],
        *,
        valid: Mapping[str, np.ndarray] | None = None,
    ) -> None: ...

    def merge(self) -> dict[str, xr.DataArray]: ...  # drain completed parents

    @property
    def pending(self) -> dict[str, int]: ...  # include parents never started

    def finish(self) -> None: ...  # reject incomplete full jobs
```

`parents` contains lazy native rasters providing target shape and exact original
coordinates/metadata. Pixel placement comes from the reference. Standalone
assembly resolves persisted parent assets into this same mapping; it does not
need a live `Tiles`. A reference alone does not retain every parent coordinate,
time label, variable attribute, or padding recipe needed to reproduce inputs.

The first dense contract requires full-tile outputs on the prepared input grid:
last two dimensions match recorded `height, width`. Reject cropped/downsampled
results instead of guessing a transform. Padding contributions are clipped in
parent pixel space. Optional validity masks are spatial boolean arrays; finite
checks also exclude invalid prediction values. Default finite checks cannot
recover input nodata after `nan_to_num`, so consumers preserve validity before
that conversion. Policy for multiple input groups is task-owned and explicit.

Segmentation blends continuous logits and decodes after parent assembly.
Regression blends values and preserves variable identity and units. Never blend
class IDs. Consumers attach output class/variable coordinates and units from the
model; the geodata accumulator must not copy input-band meanings onto predicted
bands. Validation reads ground truth from the original prepared parent:

```python
# Supervised evaluation assembles only logits; targets remain original.
merger.add(
    dict(zip(ids, logits.detach().float().cpu().numpy(), strict=True)),
    valid=dict(zip(ids, valid.cpu().numpy(), strict=True)),
)
for parent_id, logits in merger.merge().items():
    target = dataset.parents[parent_id][dataset.target].dataset
    # Convert target through the same nodata/ignore-index rule as tile labels.
    # Assert parent grid and shape agree, then score this completed parent once.
```

This removes the target merger and its averaging/rounding of categorical labels.
Evaluation keeps full-parent scoring and raw tile `logits`/`label` callback
outputs. A full prediction job calls `finish`; Lightning's deliberately limited
validation/sanity runs may leave parents pending and must not score them.

## Output families and workflow choices

| Output | Per-tile interpretation | Parent result |
| --- | --- | --- |
| Semantic segmentation | Raw class logits | Blend logits, then decode raster |
| Pixelwise regression | Values on the declared grid | Blend values, retain units |
| Reconstruction / MAE | Reconstructed pixels and explicit mask policy | Dense assembly only if the output grid and coverage are defined |
| Detection | Model-specific decode to tile-local boxes | Translate, clip, deduplicate objects per parent |
| Classification / embeddings | Scores or feature vectors | Keyed sample records; aggregation is explicit |
| Instance / panoptic segmentation | Masks plus object identities | Object association and mask policy; not ordinary dense blending |

There is shared model execution, but no universal merging rule. IDs give every
consumer location without defining its result semantics. A successful tile with
zero objects must appear in completion records even though it adds no boxes.

Three workflow options remain concrete candidates:

```python
# A: complete jobs by output family (initial recommendation).
predict_dense(...)
predict_detection(...)
predict_classification(...)

# Maintainer for A: each flow owns its actual completion and result policy.
@flow
def predict_dense(model, dataset, output):
    # Read inputs, run the model, feed selected tensors to TileMerger.
    # Drain completed parents, decode, persist, then require finish().
    ...

# B: independently useful persisted inference and assembly jobs.
raw = predict_tiles(...)
rasters = assemble_dense(raw, ...)
objects = assemble_detections(raw, ...)

# Maintainer for B: raw records and completion are durable operation boundaries.
@flow
def assemble_detections(products, reference, output):
    # Read products/reference, check completion, map boxes, deduplicate, write.
    ...

# C: general orchestrator for a mixed model (later, explicit output consumers).
# Maintainer: execute model(**inputs) once per batch, then pass each selected
# native result plus ids to its concrete consumer. Do not infer tasks from rank.
```

A uses fewer intermediate formats and fits independently runnable completed
jobs. B earns its extra persistence only when resume or reuse needs it. C must
avoid invoking the shared encoder separately for each output. We can reach C
without a registry after two concrete consumers establish what repeats.
These sketches do not introduce workflow signatures, CLI flags, or YAML fields
in the foundation change. Model transforms must be exercised in the first real
flow; the previous full-flow probe used an empty transform recipe.

## Ownership and boundaries

- `geodata/transform/tiling.py`: lazy cutting, reference geometry/pixel windows,
  and numeric dense placement. No torch or task decoding.
- `model/chain/`: native terminal result composition. Model-specific detection
  decoding belongs with the actual model implementation, before spatial assembly.
- `ml/<head>/<method>/`: dataset targets, augmentation, losses, scoring,
  Lightning prediction. Keep `forward()` tensor-only for supervised segmentation.
- `workflow/flows/`: completed jobs; `workflow/tasks/`: reusable operations with
  meaningful orchestration boundaries. Avoid one flow per head class.

Keep model-owned `FramesSpec`/`TilesSpec`, Kornia random training crops, existing
release ownership, and the editable development setup. No compatibility aliases,
mandatory saved tile rasters, wheel build, publishing, or workspace edits.

## Evidence and unresolved gates

The [full-flow probe](../probes/2026-10-04-predict-report.md) verified 36 shuffled
tiles, geographic/pixel-only persistence, dense placement, and native detection
records. It established neither a selected backend nor detection quality.

The [terminal-output probe](../probes/2026-10-04-terminal-outputs.py) now imports
the actual library implementation. It passed single tensor/list results, mixed
tensor/list results, two tensor heads, encoder-only behavior, empty detections,
gradient propagation, and outer-type rejection. Regression tests cover runtime
annotations including unsupported members of optional unions.

The [backend comparison](../probes/2026-10-04-merger-comparison-report.md)
recommends keeping Tiler window generation and replacing live-cut accumulation
with reference-driven placement. It verifies validity, completion, coordinates,
persisted reorder, and actual accumulator memory; buffers remain full-parent.
The [migration plan](../plans/2026-10-04-reference-merger.md) now replaces
production dataset positions with IDs and reads original parent labels.
Standalone weighting receives the original recipe through `overlap`; supervised
batches carry an explicit fourth validity tensor before NaN conversion. Before a detection flow: settle the
chosen detector's box convention/decoder and all-empty completion persistence.
MAE training, new regression/detection training methods, and HoloViz EDA follow
these foundations under their own reviewed designs.
