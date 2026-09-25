# Geodata review for the workflow redesign

Date: 2026-09-22. Scope: recommendations only; no geodata source changes.

The workflow implementation discussed during this review was removed on
2026-09-23. These geodata observations are retained for review, not as an
implementation plan. Current workflow boundaries are recorded in
[`workflow/README.md`](../../../src/geosave_engine/workflow/README.md).

The native xarray/accessor approach is the right foundation. Keep explicit
reprojection, masking, unpacking and tensor conversion, and keep Lightning and
Prefect out of the geodata core. The changes worth making concern a few concrete
boundaries, not a replacement raster framework.

## Before implementing persistent workflow stages

### Reject unsupported writer destinations

[`utils/io/zarr.py:254`](../../../src/geosave_engine/geodata/utils/io/zarr.py#L254)
converts the destination to `Path` before deciding how to write. Passing an S3
URI therefore becomes a relative local `s3:/...` path rather than an object-store
write. Reads accept URIs, so the asymmetry is easy to miss.

Proposed change: explicitly reject URI destinations in local writers. Add remote
storage only when both the underlying writer and publication semantics support
it. Do not introduce a storage backend hierarchy just to make the signatures
look symmetrical.

Verification for that change: a URI must fail before creating a local directory;
local Zarr round-trip and variable-order tests must continue to pass.

### Separate structural raster validation from model requirements

[`core/base.py`](../../../src/geosave_engine/geodata/core/base.py) deliberately
allows a missing geobox and GCP grids. That is useful general xarray behavior.
[`core/raster.py:_stacked`](../../../src/geosave_engine/geodata/core/raster.py#L774)
checks conversion constraints, but it is not an ingestion validator.

Proposed change: add small model-independent checks for locatable regular grids,
spatial coordinates, variable dimensions, and internally consistent nodata and
packing metadata. These should accept native xarray objects and ordinary
arguments, perform no repairs, and avoid reading pixel arrays. ML then compares
those objects against the model's requirements for product semantics, channels
and units. Do not import model-specific requirements into geodata or reject every unreferenced xarray object
globally. Chunked pixel-quality validation should be a separate explicit step.

Verification: malformed metadata fails without computing a Dask array; valid
GeoTIFF, NetCDF and Zarr data preserve their metadata when reopened.

### Keep write completion separate from format encoding

[`utils/io/zarr.py:write`](../../../src/geosave_engine/geodata/utils/io/zarr.py#L216)
writes directly to its destination, and `overwrite=True` uses xarray's replacement
mode. That is a normal low-level writer contract, but does not protect an earlier
valid workflow result from a failed replacement. A delayed write is also not a
completed result.

Proposed change: keep format encoding in geodata. Implement stage-and-publish
policy at workflow persistence boundaries, using unique staging locations and an
explicit collision policy. Return a reference only after compute and completion
checks succeed. Object stores need their own publication protocol; a local rename
is not portable atomicity. No geodata writer should require an active Prefect
transaction.

[`pipeline/manifest.py:save`](../../../src/geosave_engine/geodata/pipeline/manifest.py#L208)
already stages a manifest, but uses one fixed staging filename and has no
multi-writer merge policy. Keep updates single-writer in a flow; consider unique
staging names and locking only if multiple writers are actually introduced.

## Before implementing shared training/prediction encoding

### Place model sample assembly in ML

[`datasets/tiles.py:TileDataset`](../../../src/geosave_engine/geodata/datasets/tiles.py#L75)
emits `image`, `index`, and optional `model_context`, and converts through
`.gs.to_tensor(dtype=...)`. This is a useful raster adapter, but it does not apply
the shared model data specification. It cannot by itself guarantee equal training
and inference tensors or assemble named text/prompt inputs.

Proposed change: keep `Tiles` and reconstruction in geodata. Put the model-aware
dataset/sample adapter under `ml.data` and have both Lightning data loading and
inference share the same encoding behavior. Reuse the existing native tensor conversion;
normalization, model argument binding, and context extraction belong to that ML
adapter. Decide whether the current thin `TileDataset` remains useful after the
adapter exists; avoid a second implementation of encoding in it.

Verification: the same prepared tile produces equal named tensors and context in
training and inference with augmentation disabled. Cover packed/nodata imagery,
boolean masks, time axes, variable order and conversion overflow. Prompt geometry
and label augmentation must move together when those modalities are implemented.

### Make reconstruction memory limits real

[`transform/tiling.py:TileMerger`](../../../src/geosave_engine/geodata/transform/tiling.py#L602)
constructs a `tiler.Merger` over an entire scene. Tile reads are lazy, but the
output accumulator is scene-sized. Large rasters or many output classes can still
exhaust memory.

Proposed change: first estimate padded accumulator, weight and temporary output
memory before starting inference and enforce `max_scene_bytes` in the ML runner.
Release a scene when it completes. Partitioned or disk-backed reconstruction is
a later change justified by actual workload sizes, not a prerequisite for this
skeleton. Logits must still be reconstructed before argmax or thresholding.

Verification: a scene that exceeds the limit fails before accumulator allocation;
overlapping tiles reproduce direct prediction on a small known raster.

## Before enabling acquisition caching

### Expose resolved STAC items without repeating a live search

[`stac/source.py:load`](../../../src/geosave_engine/geodata/stac/source.py#L329)
currently combines search, lazy loading, metadata assembly and CRS stamping.
It preserves published digital numbers and delegates to odc-stac, which are
useful properties to retain. The combined operation makes it harder to freeze
source identity before deciding whether cached downstream work can be reused.

Proposed change: expose a search/resolve operation returning native pystac items,
and a load operation accepting those resolved items. The convenient `load(anchor)`
can compose those two operations. Record collection/item/asset version identity
without treating expiring signed URL tokens as data versions. Apply rate limits
to actual catalog requests and deferred asset reads; limiting task count alone
does not impose a requests-per-second limit.

Verification: loading resolved items does not search again; changing resolved
asset identity invalidates downstream reuse; lazy raster reads retain provider
authentication until materialization.

## Suggested order

1. Writer destination guards and structural raw validation.
2. Completed-write publication plus prepared-data metadata round trips.
3. Shared ML sample encoding and equality checks through Lightning and prediction.
4. Memory-bounded reconstruction.
5. Resolved STAC acquisition and cache/asset policies.

The existing native data objects, format readers, geospatial transforms and
tiling helpers remain the building blocks throughout.
