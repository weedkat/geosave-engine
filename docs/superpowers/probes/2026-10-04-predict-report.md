# Prediction design probe

Run date: 2026-10-04. Status: completed experiment, not a production workflow or
selected production merger. The script now uses the implemented public tile
reference API and native terminal model outputs.

Source: [2026-10-04-predict.py](2026-10-04-predict.py).
Latest artifact directory: `/tmp/geosave-full-predict-sdrlo2em`.
Machine-readable results: `report.json` in that directory.

## What ran

One actual local Prefect flow, `prediction-design-probe`, with four tasks:

1. Read and validate prepared input rasters through `ModelSpec`, cut them with
   current `Tiles`, and persist a reference GeoDataFrame.
2. Discard the cut, reconstruct samples from recorded pixel windows, run small
   native models in shuffled batches, and write raw prediction products.
3. Read the persisted reference and products, assemble dense predictions, decode
   segmentation, and write completed rasters.
4. Associate detection records with the reference, translate boxes to parent
   pixels, clip them, suppress duplicates by class, project polygons where
   georeferencing exists, and persist detection records.

The local Prefect database recorded the flow and all four tasks as COMPLETED.
The first sandboxed run could not create sockets; the experiment then ran with
approved local socket access. No STAC request, model download, training,
publishing, or library workflow implementation was involved.

## Fixtures and native model composition

Three 12 by 16 RGB rasters: two georeferenced parents sharing the same UTM grid,
and one pixel-only parent. Each has one invalid pixel. They produce 36 tiles of
8 by 8 pixels with overlap 4 and reflected padding.

```python
dense = ModelChain(
    features=TinyEncoder(),
    cover=SegmentationHead(...),
    amount=RegressionHead(...),
)
classifier = ModelChain(features=TinyEncoder(), category=ClassificationHead(...))
detector = ModelChain(objects=StructuredDetector())
```

The dense encoder is a learned 1 by 1 convolution. Its pointwise computation
provides an exact full-parent placement oracle. Arbitrary stage names and two
terminal tensor outputs worked without assuming `model.head` exists.

`StructuredDetector` contains native Torchvision Faster R-CNN with a tiny
backbone and ROI head: **1,529 parameters**. It emits native per-image dictionaries
containing boxes, labels, and scores through `chain_step(head=True)`. A single
detector chain returns that native list unchanged.

These models are untrained. Predictions test execution and representation, not
geospatial accuracy, detection quality, or useful classification performance.

## Verified behavior

| Concern | Result |
| --- | --- |
| Persistent association | Shuffled batches and reordered raw asset rows resolved through reference IDs |
| Dense multihead | Segmentation logits and regression values returned under their stage names |
| Dense placement | Averaging and Hann assembly matched direct pointwise inference within absolute tolerance 1e-6 |
| Validity | Each parent's 191 valid pixels had positive accumulated weights; the invalid pixel remained nodata |
| Padding | Signed offsets clipped contributions to the original parent extent |
| Persistence | Geographic label COGs and pixel-only label NetCDF round-tripped; regression NetCDF retained values and units |
| Classification | 36 unique keyed records; pixel-only footprints stayed null |
| Native detections | 288 raw detection records became 191 records after clipping and class-aware suppression |
| Detection coordinates | Native resizing returned boxes in original 8 by 8 tile coordinates before parent translation |
| Cross-tile duplicate test | Different local boxes from neighboring tiles mapped to the same parent box; same-class duplicate suppressed, different class retained |
| Empty detection operations | Native detector returned empty boxes at threshold 1.0; empty NMS and empty table construction succeeded |
| Completion | Duplicate raw raster records and a wholly missing parent's raster records were rejected |

Detection completion is recorded separately from detection rows, since a tile
with zero detections is still a completed prediction. The controlled NMS test
does not establish that NMS is the right association policy for every detector,
instance mask, or partially clipped object.

## Model flexibility findings

1. Multiple tensor terminal heads work. Single classification output works.
2. A named structured detector output works by itself.
3. `chain_step(head=True)` supports a native `list[dict[str, Tensor]]` return type.
4. A chain combining terminal segmentation logits and a terminal detector returns
   both outputs under their stage names. Named intermediate outputs remain
   internal when terminal results exist.
5. GeoSave's existing `DetectionHead` produces raw `(B, cells, 4 + classes)`
   values. This experiment checked its `(2, 64, 5)` output but did not reinterpret
   those values as boxes. Its output parameterization and decoding need a design.

The earlier run exposed the tensor-only terminal limitation and named structured
output exclusion. The implementation now generalizes terminal declarations;
`assess_model_contracts()` verifies the changed native and mixed behavior. This
does not supply arbitrary task decoding/assembly. Native output containers remain
ordinary PyTorch results; no universal prediction wrapper was added.

## Workflow options grounded in the experiment

### One general completed prediction job

```python
# Illustrative candidate, not implemented.
predict(model, reference, task="segmentation", output=...)

# Maintainer: one job must choose task-specific result assembly.
if task == "segmentation":
    assemble_dense(...)
elif task == "detection":
    assemble_detections(...)
```

A general orchestration function is possible: the probe itself ran all tasks
in one flow. However, an accumulating task switch combines unrelated policies
and gets ambiguous for models returning several output families. A scalar task
selector does not solve mixed-output routing or native decoding semantics.

### Completed jobs by output family

```python
# Illustrative candidates, not implemented.
predict_dense(model, reference, output=...)       # segmentation + regression
predict_detection(model, reference, output=...)
predict_classification(model, reference, output=...)

# Maintainer: each job assembles its actual result.
@flow
def predict_detection(model, reference, output):
    for predictions, ids in infer_batches(...):
        record_detections(predictions, ids, ...)
    return assemble_detections(...)
```

Segmentation and pixelwise regression share dense placement/blending. Detection
requires geometry conversion and object deduplication; classification produces
sample records with no automatic scene aggregation. Reusable preparation and
batch execution can remain ordinary Python operations.

This is the current recommendation for initial complete jobs, subject to the
model-output contract. It is not one flow for every neural-network head class.
Instance segmentation combines masks and object identities, so output family
and assembly semantics matter more than a head name alone.

### Shared inference, then independently runnable assembly

```python
# Illustrative candidate, not implemented.
raw = predict_tiles(model, reference, output=...)
rasters = assemble_dense(raw, ...)
objects = assemble_detections(raw, ...)
```

This supports reusable intermediates and a future general orchestrator for a
mixed model without repeated encoder execution. Persistence is justified when
raw results are independently useful. Its storage representation differs by
output, and validation should also support an in-memory path.

Do not freeze a generic dispatcher, output registry, new model-spec field, or
mandatory intermediate raster store based on this experiment. Native terminal
outputs are now implemented; production assembly still requires two concrete
consumers before extracting their shared inference code.

## Limits and checks

Reference preparation now calls the implemented `Tiles.reference()` API without
private cutting layout access. The accumulator remains disposable experiment
code. Accumulation allocates full-parent NumPy buffers and
uses an explicit Python tile loop. It is not a bounded-memory backend choice.

Other unverified concerns include convolutional seam quality, attention context,
multiple processes or devices, rotated output grids, output crops/stride,
production resume semantics, model release round trips, and large-scene memory.
STAC ingestion and Lightning training paths remain unchanged.

Checks performed: successful local Prefect flow/task execution, assertions in
the reproducible probe, and Ruff for the probe. The updated foundation
preservation run passed **337 tests, 9 deselected**, including chain, head,
release, reference, dataset, and segmentation coverage. Reproduce with:

```bash
PREFECT_HOME=/tmp/geosave-predict-probe/prefect \
PREFECT_SERVER_ANALYTICS_ENABLED=false \
MPLCONFIGDIR=/tmp/geosave-reference-mpl \
uv run --no-sync python docs/superpowers/probes/2026-10-04-predict.py
```

Sources:

- [Torchvision native detection inputs and outputs](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
- [Torchvision class-aware NMS](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html)
- [Prefect flow execution](https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run)
