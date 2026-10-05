# Indexed tile reference

> Pixel wrapper APIs in this historical design are superseded by
> [native pixel tiling and a geospatial reference](2026-10-05-pixel-tiling-reference-design.md).

Status: indexed reference implemented, 2026-10-04. This document records its
schema and preservation audit. The reference-driven production merger and
keyed datasets migrated on 2026-10-05 under the separate implementation plan.

The later [prediction foundations refinement](2026-10-04-prediction-foundations-design.md)
defines native terminal outputs, the target dense assembly interface, and
original-parent target scoring. Its [implementation sequence](../plans/2026-10-04-prediction-foundations.md)
coordinates this reference slice with model results and consumer cleanup. This
document remains the detailed reference schema and preservation audit.

The pivot is from routing through a live cut to routing through reference
records. Adding a table alone does not complete it: dataset keys, scene/frame
identity, metric assembly, examples, and obsolete execution paths must migrate
together after a replacement assembler is selected.

## Caller skeleton

```python
# Implemented: one spatial cut and its native reference table.
tiles = spec.tiles.cut([scene_a, scene_b])
reference = tiles.reference(parent_ids=["scene-a", "scene-b"])
lookup = reference.set_index("id", drop=False, verify_integrity=True)

# Proposed dataset behavior. position is only a DataLoader position.
inputs = read_inputs(spec, tiles[position])
return inputs, reference.iloc[position]["id"]

# Lightning already supports carrying a key beside raw model outputs.
inputs, ids = batch
predictions = model(**inputs)
rows = lookup.loc[list(ids)]
```

The dataset returns a persistent row ID, not a GeoDataFrame position. String
IDs use ordinary PyTorch collation and become a list beside the batched tensors.
The native model receives only its tensor inputs. A prediction writer may use
the IDs from the original batch, or a prediction step may return them beside
the outputs when predictions leave the callback lifecycle.

## Intent and established requirements

- A prediction key must recover its reference record after shuffling,
  filtering, reordering, and GeoParquet reload.
- One row describes one tile of one prepared parent raster/frame.
- One reference GeoDataFrame can describe several parents. The merger must
  keep parents distinct even when their geometry is identical.
- Georeferenced tiles reuse existing `proj:shape`, `proj:transform`, and
  `proj:code` / `proj:wkt2` fields. This choice was explicitly selected.
- Rasters without georeferencing remain supported. Their CRS, transform, and
  geographic geometry must not be fabricated.
- Keep native pandas/GeoPandas, xarray, and Lightning objects. No new record,
  prediction, or training wrapper is required.
- Keep `FramesSpec`, `TilesSpec`, and current model-spec tiling fields. A frame
  participating in a cut needs its own parent identity; source scene identity
  alone cannot distinguish overlapping temporal frames.

## Reference schema for both raster kinds

| Field | Purpose |
| --- | --- |
| `id` | Unique tile key, preserved as a column across persistence |
| `parent_id` | Identity of the prepared raster/frame being reconstructed |
| `row_off`, `col_off` | Signed pixel offsets relative to that parent's grid |
| `height`, `width` | Spatial shape of the cut input tile; exact-grid outputs reuse it |
| `geometry` | Geographic footprint when available; otherwise null |
| `proj:shape` | Exact georeferenced tile shape; otherwise null |
| `proj:transform` | Exact georeferenced tile affine transform; otherwise null |
| `proj:code`, `proj:wkt2` | Pixel-grid CRS by code or WKT; otherwise null |

The common pixel-window columns support
non-georeferenced rasters. For georeferenced tiles they can be derived from
the parent and tile grids, but storing them also gives both kinds the same
destination-window lookup. Their shape must agree with `proj:shape` when present.
This duplication serves two coordinate systems: parent pixel placement and
world georeferencing. It is not a second independently configured recipe.

Geometry is expressed in EPSG:4326 for referenced rows, following the existing
catalog convention. A mixed frame's CRS describes its non-null geometries;
it does not claim that a null-geometry row has georeferencing. An entirely
unreferenced frame has null geometry and no CRS.

Use native `geodata.io.geoparquet.write/read` for the reference. The existing
`.gs` GeoDataFrame accessor requires a CRS, so an all-unreferenced frame must
not be forced through it. The reference has no `assets` column initially and
is not presented as a STAC item catalog. Saved predictions can be registered
later through the existing asset APIs when the persistence contract is designed.

The prepared parent rasters supply target dimensions and, when present, target
georeferencing. The reference is the one prediction-to-record lookup table;
it does not replace the original parent raster metadata. Independent persisted
assembly will need durable parent metadata or an explicit parent asset link.

## Maintainer skeleton

```python
# Implemented in geodata/transform/tiling.py.
def reference(self, parent_ids: Sequence[str]) -> gpd.GeoDataFrame: ...
```

The builder reads layout and raster metadata only. It must not compute source
pixels. Its rows follow the existing `Tiles` sample order; workers then read
the ID associated with the same tile position.

Parent IDs must be non-empty, unique, and match the number of prepared parents.
Generated IDs are `f"{parent_id}/tile-{local_tile_id}"`, with local
IDs coming from the current cut. They are opaque lookup keys, never parsed to
recover placement. They identify the persisted reference snapshot; they do not
promise stability across a changed tiling recipe. Changing source order must
not change the IDs of a parent whose own cut is unchanged.

Offsets include the existing outer frame: subtract the before-padding from
layout offsets. Negative offsets are legitimate. The merger clips contribution
windows to the target while keeping tile-relative weighting coordinates intact.

No change to `Tiles.__getitem__`, `locate`, `merger`, or `TileDataset` is required
to expose this reference. Training-path migration is implemented by the
[reference merger plan](../plans/2026-10-04-reference-merger.md).

## Current consumers and cleanup audit

Inspected 2026-10-04. These are active consumers, not dead code merely because
the design is changing.

| Current code | Assumption to migrate | Required preservation |
| --- | --- | --- |
| `geodata/transform/tiling.py`: `TileMerger.add`, `merge`, `_build_merger` | Routes numbers through `Tiles.locate`; reads `_rasters` and `_frame`; uses the live tiler to count completion | Correct placement, edge weights, duplicate rejection, batch validation before mutation, completion, leading dimensions, dtype |
| `ml/datasets/tiles.py`: `TileDataset` | Returns a loader-local integer and documents reconstruction through the original cut | Lazy reads, input ordering, contextual inputs, tensor-only native model calls |
| `ml/segmentation/supervised/data.py`: `Dataset._cut`, `tiles`, `merger`, `__getitem__` | Flattens scene/frame stacks without retaining source identity; constructs merger from the worker-local cut | Capture manifest/frame identity before flattening; retain process-local readers and clean pickle/reopen behavior |
| `ml/segmentation/supervised/module.py`: `_merge` | Calls `index.tolist`; creates two live-cut mergers; pairs logits and targets by raster ordinal | Parent-key pairing, full-parent validation/test metrics, incomplete-epoch behavior, raw logits before interpretation |
| `ml/segmentation/supervised/data.py`: `on_after_batch_transfer` | Type annotations assume tensor indices; augmentation changes training pixels while passing identity through | IDs survive collation/device transfer; training augmentation does not make a key describe an augmented geographic footprint |
| `model/README.md` and source docstrings | Prediction examples reconstruct through `tiles.merger()` | Runnable examples using the actual selected contract, including explicit transforms |
| Mirrored tiling, dataset, supervised-module, and encoder-context tests | Some assert numbering; others prove valuable behavior | Rewrite representation assertions while preserving behavior coverage |

The new reference must identify prepared temporal frames before the dataset
discards their relationship to manifest rows. Rebuilding a cut in another worker
must reproduce the same keys and windows. Different splits or loaders may reuse
local IDs; assembly must include its owning reference/loader context rather than
silently combining them.

Preserve the original parent's coordinate labels when producing completed
rasters, not only an equivalent affine transform. Existing tiling tests cover
exact coordinate restoration, named leading dimensions, and lazy padding; these
remain requirements for replacement assembly.

Completion must consider all expected reference IDs, including a parent for
which no prediction ever arrived. The current `pending` property only counts
started parents. That limitation is not the intended persisted-reference
completion contract.

Ground-truth labels need their own assembly policy. The current implementation
uses a numeric target merger followed by rounding. Retain correct results for
identical overlapping labels, but explicitly test conflicting and invalid target
pixels before carrying this policy into a reference-driven assembler. Blending
raw logits does not justify averaging categorical labels.

`DensePredictionLogger` and `ThresholdCalibrator` currently consume the
validation/test step's `logits`/`label` output, which contains tile tensors while
metrics are assembled from completed parents. Preserve that output contract
during the tile migration. A later change to scene-level calibration/logging
needs explicit tests and must not be smuggled into ID cleanup. Moving threshold
ownership into the model head remains a separate planned concern.

`FramesSpec`, `TilesSpec`, `read_inputs`, existing raster padding/metadata helpers,
and native `mosaic`, `concat`, and band `merge` are not obsolete merely because
the merger changes. Evaluate each module by the behavior it owns. In particular,
`Tiles.locate` may still be needed internally by the cutting implementation even
after external merger routing stops calling it.

### Historical proposals

The 2026-10-03 head-owned prediction draft proposes removing spec tiling,
rebuilding a tiler from `TileLayout`, head-owned file reading/stitching, and a
`model.head` assumption. Those parts are superseded proposals, not current
requirements. Its named padding defect has already been addressed in the live
tiling implementation; do not schedule it again without a failing reproduction.
Keep historical evidence but label incompatible proposals clearly.

### Migration completion gate

After the backend comparison, the follow-up implementation plan must:

1. Migrate general prediction and supervised validation/test in one coherent
   slice, including types, callbacks' existing contracts, templates, and examples.
2. Prove reconstruction from a persisted reference plus explicit parent metadata
   after discarding the original `Tiles`. A table that only maps new IDs back to
   old positional indices does not establish independent assembly.
3. Port lazy-read, worker, spatial/temporal context, completed-scene metric,
   nodata, duplicate/incomplete-result, and discrete-target tests.
4. Remove obsolete merger routing, helpers, imports, and tests after every active
   consumer is migrated. Do not retain two production paths or alpha aliases.
5. Remove the `tiler` dependency only if the selected cutting and accumulation
   implementation no longer uses it. Backend selection and cleanup are separate
   from deciding whether the `Tiles` public name remains useful.

Temporary coexistence while testing the candidate reference is a migration
stage, not the final architecture. No production path is removed by this draft.

## Geometry and placement without georeferencing

```python
row = lookup.loc[tile_id]

# Both kinds have a destination window in parent pixel space.
window = Window(row.col_off, row.row_off, row.width, row.height)

# Only referenced records reconstruct a geographic grid.
if row["proj:transform"] is not None:
    grid = GeoBox(
        tuple(row["proj:shape"]),
        Affine(*row["proj:transform"]),
        row["proj:code"] or row["proj:wkt2"],
    )
```

These are illustrative scalar values before pandas' null normalization. An
implementation must normalize missing code/WKT and projection cells explicitly;
it must not rely on the truth value of `pd.NA` or a NumPy array from Parquet.

A pixel-only output remains an unreferenced xarray raster. A geographic output
retains its actual geobox. Neither path silently assigns EPSG:4326 or an identity
affine to an unreferenced raster. This does not add support for a DataTree whose
groups lack the shared grid currently required by `Tiles`.

## The merger comparison

Do not select a backend from the row schema alone. Compare three candidates:

1. Existing `Tiler` plus `Merger`, adapted only if the indexed reference can
   recover routing without making tile IDs a hidden layout dependency.
2. Keep `Tiler` for window generation and use pixel-window accumulation driven
   by the reference. Existing NumPy/Dask/Rasterio operations remain available.
3. Native window generation plus window-driven accumulation, replacing `tiler`
   only if coverage, weighting, validity, and maintenance justify it.

For each candidate, show caller and maintainer code, then measure the same
georeferenced and pixel-only fixtures. Report source coverage, edge behavior,
invalid-pixel handling, intermediate storage, and allocated accumulator memory.
New window loops are not evidence that the entire tiler can be replaced cheaply.

Placement must be independent of input order and separate from blending policy.
At minimum compare plain averaging and the currently used Hann weighting.
Invalid values contribute neither prediction values nor weights. Valid zero
logits remain valid. Incomplete and duplicate results must remain detectable.

Do not force detection, classification, or embeddings through dense blending.
The same row ID can recover their sample context, but their assembly differs.
Keep segmentation decoding after continuous raster assembly. Detection decoding
must establish tile-local object coordinates before mapping/association. There
is no universal ordering of interpretation and assembly across output families.

## Scope and unresolved behavior

The implemented reference slice keeps the existing tiler and training path
operational. The [comparison report](../probes/2026-10-04-merger-comparison-report.md)
recommends Tiler windows plus reference-driven accumulation. Production
validation/test now assemble logits by reference ID and read original targets.

Still to settle through tests and examples:

- Head outputs that crop, downsample, or change the input grid.
- Input validity and halo ownership, especially at padded edges.
- Completion, duplicate handling, and bounded-memory assembly.
- Durable parent metadata for independent assembly after a process exits.
- Saved output format for pixel-only rasters; GeoTIFF is not mandatory.

## Evidence recorded so far

On 2026-10-04, throwaway probes verified:

- Forty exact tile grids across two parents recovered from existing projection
  fields after a native GeoParquet round trip and row shuffle.
- Forty tile grids recovered from parent grid plus signed pixel windows,
  including negative padding offsets.
- Persistent IDs retained prediction association through shuffled batches and
  reordered Parquet rows, including identical footprints in different scenes.
- Lightning's batch prediction writer received the original spatial context
  while `predict_step` returned only a tensor.
- Pixel-only and mixed reference GeoDataFrames round-tripped through native
  GeoParquet I/O with null geometries preserved. Existing merging reconstructed
  pixels and preserved absent georeferencing for pixel-only outputs.
- `tests/model/spec/test_cuts.py` and `tests/ml/datasets/test_tiles.py`:
  14 passed.

These probes establish association and metadata feasibility, not a final merger
or its memory, seam-quality, validity, and persistence guarantees.

## End-to-end prediction experiment

The later [prediction-flow probe](../probes/2026-10-04-predict-report.md) ran a
real local Prefect flow with native small models, persisted references/products,
geographic and pixel-only parents, and task-specific assembly. Dense results
matched direct pointwise prediction; classification and detection produced
records. The disposable script now calls the implemented public reference API.

The initial run exposed a tensor-only terminal limitation. Native terminal
containers and mixed tensor/detection results are now supported and tested in
production ModelChain. The combined foundation preservation run passed
337 tests, 9 deselected. Completed jobs by output family remain candidates;
reusable inference plus task-specific assembly remains the common direction.
