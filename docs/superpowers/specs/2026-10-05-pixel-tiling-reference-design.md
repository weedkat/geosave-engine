# Native pixel tiling and a geospatial reference

Status: pixel ownership implemented on 2026-10-05. The shared Dataset and
context portions are superseded by [row-based samples](2026-10-05-row-based-samples-design.md). This
design replaces the pixel ownership in the earlier indexed-reference and
reference-merger designs; their persistent identity and exact-grid decisions
remain useful.

## Caller first

The metadata factory returns a native GeoDataFrame:

```python
from tiler import Merger, Tiler
from geosave_engine.geodata.core.vector import GeoVector

# Original prepared scenes or temporal frames, keyed by persistent parent ID.
parents = {"scene-a/frame-0": scene_a, "scene-b/frame-0": scene_b}
layouts = {
    key: Tiler(
        data_shape=(parent.sizes["y"], parent.sizes["x"]),
        tile_shape=(256, 256), overlap=32, mode="reflect",
    )
    for key, parent in parents.items()
}

# Reads metadata and layout, never pixels.
reference = GeoVector.from_tiles(parents, layouts)
reference = reference.set_index("id", drop=False, verify_integrity=True)

mergers = {
    key: Merger(layout, logits=class_count, window="boxcar", save_visits=False)
    for key, layout in layouts.items()
}

# Predictions may arrive in any order. IDs travel beside model tensors.
for sample_id, prediction in predictions:
    row = reference.loc[sample_id]
    mergers[row.parent_id].add(int(row.tile_id), prediction)

pixels = mergers["scene-a/frame-0"].merge()
```

This is the native [Tiler/Merger interface](https://github.com/the-lay/tiler),
with one metadata lookup before adding a prediction. Geometry is not used to
reconstruct a Tiler, decide overlap, or place pixels.

## One reference table

Each row describes one spatial tile of one prepared parent/frame:

| Field | Meaning |
| --- | --- |
| `id` | Persisted sample ID, unique within this reference snapshot |
| `parent_id` | Prepared scene/frame ID; selects its parent and native layout |
| `tile_id` | Native Tiler integer ID, scoped to that parent |
| `row_off`, `col_off` | Signed pixel offsets from the original parent origin |
| `height`, `width` | Requested spatial tile support, including padding |
| `geometry` | Tile footprint in EPSG:4326, or null for pixel-only data |
| `proj:shape` | Exact spatial grid shape when georeferencing is available |
| `proj:transform` | Exact six affine coefficients in the tile's grid CRS |
| `proj:code`, `proj:wkt2` | Grid CRS as an EPSG code or WKT fallback |

The DataFrame's CRS describes `geometry`. Each row's projection fields describe
its raster grid. A polygon or its bounds cannot replace those fields,
particularly for rotated grids.

Keep `id` as a column when saving with `index=False`, then restore the lookup
index on read. Do not rely on row position or parse meaning out of an ID string.
Reordering parents or rows must preserve associations. A changed layout creates
a new reference snapshot; tile IDs are not identities across changed recipes.

Unreferenced rasters retain their pixel fields and IDs, with null geometry and
projection fields. A table containing only such rasters has no CRS. Use the
native GeoDataFrame directly: the current `.gs` accessor requires a CRS, so
accessor initialization is not part of this reference contract.

## Maintainer boundary

Use the existing factory location instead of introducing another wrapper:

```python
# geodata/core/vector.py — metadata-only classmethod
def from_tiles(
    cls,
    parents: Mapping[str, xr.Dataset | xr.DataArray | xr.DataTree],
    layouts: Mapping[str, Tiler],
    *,
    padding: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> gpd.GeoDataFrame:
    ...

# model/spec/cuts.py — replaces TilesSpec.cut(rasters)
def layout(self, shape: tuple[int, int]) -> Tiler:
    return Tiler(shape, self.shape, overlap=self.overlap, mode=self.mode)
```

`from_tiles` enumerates native spatial bounding boxes. It translates the
original parent's affine by the tile offsets and records the exact shape.
It owns metadata construction, not raster reads, padding, model tensors, or
accumulation. It checks genuine reference invariants: parent keys correspond
to layouts and generated sample IDs are unique.

Responsibilities:

- Native `Tiler`/`Merger`: pixel layout, native IDs, fringe padding, overlap,
  tapering, accumulation, and unpadding.
- Geodata reference factory: pixel support and optional exact geospatial grid.
- Dataset reader: bounded lazy reads of each named xarray input, retaining its
  leading band/time axes; model input conversion and existing model context.
- Training/prediction method: sample IDs, validity, completion, interpretation,
  and full-parent metrics. Raster wrapping uses the original parent's grid.

A two-dimensional Tiler callback reads a two-dimensional plane. Do not feed it
an arbitrary channel/time tensor. For named inputs with differing leading axes, the dataset flattens each
variable's leading axes into a native channel dimension. Its callback reads
that input's bounded spatial window, and native Tiler supplies fringe padding.
The reader restores the leading axes and derives tile coordinates from the
original parent. The shared reference and dense-output layout remain spatial
and independent of input channel counts.

## Padding is explicit

For a taper requiring a halo, setup uses native `calculate_padding()` and
`recalculate(data_shape=...)`. The reader supplies the padded pixel extent;
the original parents remain the metadata source. Constant halo fill matches
the native layout's constant value, including NaN. Pass those same padding widths
to `from_tiles(..., padding=...)` and `Merger.merge(extra_padding=...)`.

Subtract the before-padding widths from native bounding-box starts when
recording original-parent offsets. Negative offsets are legitimate. Footprints
describe requested support, not the valid-data mask. Derive these values from
the native layout; do not infer stride, padding, or transforms from coordinates.

One-pixel overlap is rejected for Hann, Bartlett, Barthann, Bohman, Blackman,
and overlap-tile windows because their edge weights leave uncovered seams.
Positive-edge windows retain support for one-pixel overlap.

Keep layouts and the recipe with the job. On a later process/run, recreate them
from the recorded parent shape and recipe, not from the GeoDataFrame's geometry.
The reference is sufficient to locate a tile; it is not an alternative encoding
of every native merger setting.

## Head behavior and preserved training contracts

All heads retain the same sample-ID association. Dense logits or regression
values use native pixel merging before head interpretation. Classification
stores row-associated predictions. Detection uses the row's pixel offsets and
affine to locate boxes; cross-tile detection reconciliation is a later head
policy. Do not implement those additional heads in this migration.

Validation/test continue to predict tiles and evaluate assembled parents.
Original categorical targets are never blended. Task-owned validity must
exclude invalid predictions from overlap weights and retain nodata for
uncovered output. Native merger accumulation can be used for weighted values
and coverage; a generic geodata merger must not own that policy.

Keep Lightning `forward()` returning tensors and `predict_step()` returning
the existing tensor/ID tuple. Preserve lazy worker reopening, partial-evaluation
handling, and Kornia training crops. Do not add a sampling framework, a new
workflow, or an output wrapper to accomplish this ownership change.

## Smoke evidence and migration limits

A disposable probe on 2026-10-05 used four Dask-backed parents, a small Conv2d,
native Tiler/Merger, and GeoSave GeoParquet I/O. It passed:

- Exact persisted rotated EPSG grids and a custom WKT-only CRS.
- Two parents sharing geometry with different pixels, plus pixel-only data.
- Zero pixel reads while building the reference, then bounded tile reads.
- Shuffled reference rows and two-channel predictions matching full-parent
  inference: 36 rows with boxcar, 64 rows with Hann and native halo padding.
- An entirely unreferenced GeoParquet table retaining null geometry and no CRS.

Probe: `/tmp/geosave-pixel-reference-probe/probe.py`. The subsequent implementation tests verified named multiaxis inputs,
invalid overlap and uncovered pixels, full-scene evaluation, and worker
reopening. See the implementation plan for verification results. Native Merger retains full parent-sized accumulation buffers; this
proposal makes no out-of-core merging promise.
