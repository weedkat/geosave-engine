# Model outputs and prediction assembly

> Pixel wrapper APIs in this historical design are superseded by
> [native pixel tiling and a geospatial reference](2026-10-05-pixel-tiling-reference-design.md).

Status: draft for discussion. The interfaces below are a code skeleton, not
implemented public APIs. This draft does not adopt the older tile-store proposal
or remove `Tiles.merger()`.

Update after discussion: the indexed reference and support for rasters without
georeferencing are refined in
[Indexed tile reference](2026-10-04-indexed-tile-reference-design.md).
That narrower draft takes precedence over the context-bearing batch and
georeferenced-only examples below. `blend` and the merger backend remain
candidates; they are not settled public interfaces.

## Intent

Keep native model components separate from Lightning training methods. Dense
predictions should be usable as georeferenced rasters, optionally written as
assets and indexed by GeoParquet. A later merge should not need the live `Tiles`
instance or its numbering. Validation and test must exercise the same spatial
assembly and interpretation rules without requiring intermediate disk writes.

## Package skeleton

```text
model/
  encoder/                  # reusable representations
  decoder/                  # feature or reconstruction decoders
  head/                     # output projection, output meaning, decoding
  monolith/                 # complete native models when composition adds nothing
  chain/                    # optional component composition
  spec/                     # input preparation and geospatial processing recipe
  release/                  # deployable model and recipe
ml/
  segmentation/supervised/  # existing Module and DataModule
  regression/supervised/    # future: pixelwise regression
  classification/supervised/ # future
  detection/supervised/     # future
  reconstruction/mae/       # future: native MAE inside Lightning
geodata/
  transform/composite.py    # existing first/last mosaic; proposed weighted blend
  io/                       # existing raster and GeoParquet persistence
workflow/
  tasks/                    # future: predict assets and assemble outputs
  flows/                    # future: runnable prediction jobs
```

Future folders are documentation only. Add real modules when their behavior is
implemented. No generic training base class, output wrapper, or abstract head is
needed for this skeleton. A model may be encoder-only, a composed chain, or a
complete native module. MAE reuses a native reconstruction decoder/projection;
the masking and reconstruction objective belong to its training method.

## Two assembly paths

Recommended: the same spatial blending rules with two ways to supply pixels.

```python
# Proposed in-memory caller: appropriate for validation and small predictions.
logits = blend(prediction_rasters, grid=output_grid, window="boxcar")
labels = head.decode(logits.gs.to_tensor())
```

```python
# Proposed persisted caller: appropriate for long or resumable prediction jobs.
catalog = gpd.read_parquet("run/predictions.parquet")
rows = catalog.query("scene_id == @scene_id and head == 'segmentation'")
rasters = (read_raster(run_root / asset) for asset in rows.asset)
logits = blend(rasters, grid=output_grid, window="boxcar")
labels = head.decode(logits.gs.to_tensor())
```

The saved-assets path separates GPU inference from CPU assembly, supports
inspection, and makes inference results reusable. It costs intermediate storage
and I/O. The in-memory path avoids that cost per training epoch. Their overlap,
validity, cropping, and decoding semantics must agree; parity tests are required
before claiming they simulate the same inference behavior.

The tensor model continues to accept tensors only:

```python
# Proposed prediction batch contract; spatial context is carried beside inputs.
inputs, contexts = batch
predictions = model(**inputs)
for prediction, context in zip(predictions, contexts, strict=True):
    result = raster(named_channels(prediction), context["grid"], nodata=np.nan)
    # Use result directly, or write it with the existing .gs.to_cog API.
```

`context` denotes ordinary spatial/provenance values, not a new public wrapper.
The output grid must describe the actual prediction pixels. Reusing the input
grid is valid only when the output has that same shape, extent, and alignment.
Output stride or a cropped decoder requires an explicitly derived output grid.

An index is unnecessary after the raster is georeferenced. Before that point,
an asynchronous or batched predictor must retain a reliable association between
each tensor and its spatial context. Replacing an index with a grid moves the
association; it does not eliminate it. Existing indexed batches remain valid
until this alternative is implemented and measured.

## Maintainer interfaces

```python
# Proposed geodata primitive; torch-free and independent of task semantics.
def blend(
    rasters: Iterable[xr.Dataset],
    *,
    grid: GeoBox,
    window: StitchWindow = "boxcar",
) -> xr.Dataset: ...

# Proposed semantic-head operation; independent of files and placement.
def decode(self, logits: torch.Tensor) -> torch.Tensor: ...
```

`blend` places aligned rasters using their affine transforms. It accumulates
weighted values and valid weights, normalizes covered pixels, and marks
uncovered pixels as nodata. It must operate in bounded spatial chunks for large
scenes. Reprojection and resampling remain explicit operations rather than
implicit repairs inside blending. Use dependency capabilities before writing
an equivalent spatial merger.

Keep `mosaic(method="first" | "last")` unchanged: its preference policy serves
different purposes from weighted prediction blending. Keep `TileMerger` for
current validation until the new path has equivalent behavioral coverage.

The head owns output meaning and decoding. Geodata owns spatial placement and
numeric blending. Workflow owns reading, scheduling, writing, and recording
assets. A head must not need to read a catalog or assume every chain names its
last stage `head`.

## Asset catalog

One prediction run has one catalog. Its model revision and processing recipe
must be recorded with the run so incompatible outputs are not silently mixed.

```text
scene_id | time       | head         | asset                 | geometry
scene-a  | 2026-10-04 | segmentation | tiles/prediction-a.tif | tile footprint
scene-a  | 2026-10-04 | segmentation | tiles/prediction-b.tif | tile footprint
```

`geometry` enables spatial queries; it does not replace the raster's exact
affine transform. Paths are relative to the run root. Each asset describes its
CRS, transform, dimensions, named channels, dtype, and nodata. Logit channels
must retain the same meaning and ordering across the run. Zero is a valid
logit, never a default nodata marker.

The original target grid is stored once per scene/output/time group in run
metadata, or recovered from an explicitly linked reference asset. Persist its
CRS, affine transform, width, and height; a bounding polygon alone is
insufficient. Temporal outputs must name the output timestamp or interval,
which may differ from individual input acquisitions.

The run also needs expected work/completion information to distinguish a
finished prediction from an interrupted one. Raster placement alone cannot
prove completeness or distinguish an accidental duplicate contribution from
an intentional overlap. Asset identity and run state must prevent repeated
contributions. No tile-number recipe is required to place completed assets.

## Information georeferencing cannot recover

- Scene and time grouping: two acquisitions can have exactly the same grid.
- Requested target extent: padded tiles can extend beyond the original scene.
- Valid input coverage: reflected padding and cloud/nodata pixels are not real
  observations. Preserve validity separately or mark output values invalid.
- Blend policy: averaging, center weighting, and ownership cropping differ.
- Model-specific output meaning and calibration.

For center weighting, retain the full prediction-tile footprint until blending
and crop to the target grid afterwards. Cropping first and regenerating a Hann
window changes the weights. Explicit core ownership or halo cropping requires
the corresponding valid/core footprint. Center-tapered windows must never leave
valid boundary pixels with zero accumulated weight.

## Task-specific assembly

| Output | Assembly |
| --- | --- |
| Segmentation | Blend float logits, then apply the model's decoding/calibration |
| Pixelwise regression | Blend continuous predictions, then restore units/meaning |
| Classification or scene regression | Records associated with scene/patch footprints; aggregation only if defined |
| Detection | Decode local boxes, map to world geometry, deduplicate overlapping detections |
| Instance/panoptic segmentation | Associate instances across tiles; masks alone do not settle identity |
| Reconstruction | Restore pixels from patches and blend when spatially tiled |
| Encoder embeddings | Keep their feature grid or scene identity; no mandatory raster merge |

Final segmentation labels are not suitable for numeric averaging. For detection,
one catalog footprint represents a prediction asset; individual detection
geometries belong in detection records. These are different table roles.

## Smoke evidence

On 2026-10-04, a throwaway probe used the installed dependencies and existing
GeoSave APIs to:

1. Build a georeferenced 8 by 10 raster and a small two-channel Torch model.
2. Cut 20 overlapping, padded tiles with the existing `Tiles`.
3. Write raw float predictions as COGs using each tile's geobox.
4. Round-trip a GeoParquet catalog containing asset paths and footprints.
5. Reverse asset order and merge using Rasterio's sum/count operations, cropped
   to the original grid, without any original tile indices in the catalog.
6. Verify output shape, transform, and values against direct full-scene model
   prediction with absolute tolerance 1e-5.

The probe passed. It demonstrates geospatial placement, raw-value averaging,
padding crop, and catalog independence from tile numbering. The pointwise model
does not establish seam quality for convolutional context or attention. It does
not verify Hann weighting, invalid-data handling, interrupted runs, rotated
grids, distributed writers, or bounded-memory production assembly.

Temporary artifacts: `/tmp/geosave-prediction-probe-1hllfqnl`.

Primary dependency references:

- [Rasterio merge](https://rasterio.readthedocs.io/en/stable/api/rasterio.merge.html)
- [GeoPandas spatial file persistence](https://docs.geopandas.org/en/stable/docs/user_guide/io.html)

## First implementation slice after review

Implement the dense georeferenced blend primitive and parity tests first. Then
connect segmentation decoding and persist its calibration in the native model
release. Add an asset writer/catalog seam using existing I/O only after that
behavior is sound. Regression and detection need their own output contracts;
empty training modules and a generic prediction framework are unnecessary.
