# Indexed raster reads and row-based model context

Status: implemented and verified.

Storage revision implemented: the user identified that `data=` and row-derived raster
context can describe different transformed samples. The proposed
[native sample inputs revision](2026-10-05-native-sample-inputs-design.md)
replaces the reader and persistence contracts. The old `data=` reader described
below is historical; use the current Series reader and explicit crop API.
The broader model-input revision remains deferred until native storage is sound.

The user approved implementation of the row-driven direction on 2026-10-05.
This refines the native pixel tiling design; native Tiler/Merger remain the
pixel engines. Training methods own their PyTorch Datasets.

```python
reference = GeoVector.from_layouts(parents, layouts)
reference = reference.set_index("id", drop=False)
row = reference.loc[sample_id]
tile = reference.gs.to_xarray(sample_id, data=parents[row.parent_id])
inputs = model_inputs(spec, tile.gs.rasters, row)
```

Latest user steering rejects a new reader mechanism: ordinary and tiled rows
share `GeoVector.to_xarray`. A tiled row adds a pixel boundary. `data=` supplies
already opened/prepared xarray parents; otherwise the same reader opens assets.
The result remains lazy and native xarray, with no custom tile object. Halo and
fringe extension use native padding and preserve Tiler's pixel recipe. Reflection
and wrapping pad one-dimensional indices with NumPy, avoiding Dask padding
truncation while retaining lazy pixel selection.

`GeoVector.from_layouts` replaces `from_tiles`. Rows retain exact tile grids,
native IDs, signed windows, and `raster_metadata`: metadata by prepared raster
name, containing ordered ISO timestamps and band names. Extraction reads
coordinates only, never pixel arrays. Callers may join annotations and source
asset descriptions to this native table; processed parents remain explicit.

`model/encoder/{prithvi,clay}.model_context(row, *, raster="image")` returns
model-specific tensor dictionaries. A configurable raster name selects temporal
metadata; location derives from the exact tile grid. Clay keeps constructor
wavelength/GSD defaults. Existing individual context functions also consume
rows, and the time utility reads the same row rather than raster objects.

`ModelSpec.inputs` contains raster references. `ModelSpec.context` optionally
declares a CallSpec whose runtime input is `row`. `model_context(row)` invokes
that recipe; `model_inputs(rasters, row, *, context=None)` combines the pixels
with computed or explicitly supplied cached context. Context cannot overwrite
pixel inputs. The spec stays inert on load and imports no torch. `ml/inputs.py`
owns conversion to tensors. The same function works in training and inference.

The supervised segmentation Dataset owns target conversion, validity, and
worker reopening. The shared `ml.datasets` package is removed. Detection and
classification annotations remain normal catalog columns; no new task Dataset
is introduced before that task exists. Georeference-free rows remain supported.

The row routes numpy contributions to native Merger by parent and native tile
ID. No new merger wrapper is introduced; native Merger unpads the result. Tensor conversion stays explicit in ML. Completion, invalid-pixel
coverage, interpretation, and metrics remain method-owned.

Caching is optional supplied data, not a hidden lookup or a second context
recipe. Reuse only for matching metadata, encoder recipe, and actual input
window. Augmentations currently retain the prepared tile context; recomputing
context for random crop windows remains outside this refactor.

No compatibility aliases, new dependencies, wheel builds, model downloads,
commits, publishing, or changes to unrelated working-tree edits.
