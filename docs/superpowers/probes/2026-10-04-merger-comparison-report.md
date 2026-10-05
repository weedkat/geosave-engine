# Raster merger comparison

Historical run: 2026-10-04, before the production merger migration. Source: [comparison probe](2026-10-04-merger-comparison.py).
Artifacts: `/tmp/geosave-merger-comparison-u7b7bb98/report.json`.

This is an experiment, not a second library merger. It uses the implemented
`Tiles.reference` API; production datasets and merging retain their current
contracts until the migration is reviewed.

## Caller and maintainer code

```python
# A: historical Tiler layout and live-cut TileMerger.
reference = tiles.reference(parent_ids)
outputs, memory = existing(tiles, reference, values, window="hann")

# Historical maintainer: route persistent IDs to cut-local positions.
positions = {key: position for position, key in enumerate(reference.id)}
merger = tiles.merger(window="hann")
merger.add({positions[key]: value for key, value in values.items()})
outputs = merger.merge()

# B: retain Tiler's windows; use the reference to accumulate results.
reference = tiles.reference(parent_ids)
merger = ReferenceAccumulator(reference, parents, window="hann")
merger.add(values, valid=validity)
outputs = merger.merge()
merger.finish()

# Maintainer: validate all IDs/shapes before mutation; place each contribution
# using row_off/col_off; exclude invalid values from numerator AND denominator.
# Count expected reference IDs, including parents that have received nothing.

# C: native layout arithmetic and the same reference-driven accumulation.
offsets = native_windows(parent_shape, tile_shape, overlap)
# Build matching pixel-window records, then use the same accumulator as B.
```

The native arithmetic matched existing offsets for tested layouts. Its prototype
does not replace native/lazy xarray cutting, padding modes, geospatial metadata,
or stack handling. Removing the dependency would require those preservation
tests too; matching a window formula alone is insufficient.

## Verified behavior

Three 12x16 parents: two identical geographic grids with different pixel values,
and one without georeferencing. Ten layout/window/leading-axis combinations used
8x8 tiles with overlap 0/4, and non-square 6x8 tiles with overlap 2. Assembly used
persisted, shuffled references and saved arrays in another order.

Both reference accumulation candidates matched direct pointwise full-parent
prediction and the clean current-merger baseline within absolute tolerance 1e-6.
Mean and Hann preserved padded edges, spatial coordinate labels, float32 result
dtype, geographic grids, and absent georeferencing. Named `(time, band)` leading
dimensions worked; this experiment does not establish arbitrary leading-axis
coordinate-label propagation.

The prototype rejected unknown IDs and malformed shapes before mutating a valid
preceding contribution. It rejected duplicates after completion and accounted
for wholly missing parents. Invalid pixels contributed no values or weights;
valid zeros remained covered. Explicit masks produced the same result when
invalid inputs had already been converted to zeros.

The current merger failed the partial-invalid-overlap requirement: one NaN tile
contribution contaminated valid neighbouring contributions, and Tiler converted
the resulting normalized NaN to zero. At the tested pixel, current output was
`[0, 0, 0]`, while reference accumulation retained approximately
`[0.85, 1.70, 0]`. Supporting validity in the current path would require a changed
normalization policy, not only ID lookup.

The updated [full prediction flow](2026-10-04-predict-report.md) separately tested
actual raster assets, parent asset reopening, and assembly after discarding the
cut. The comparison's saved intermediate arrays use NumPy NPZ; they do not define
a production tile-asset persistence format.

## Memory and time

One pixel-only parent, three float32 output channels, 128x128 tiles, overlap 32,
Hann. One measured run; times are illustrative, not a throughput benchmark.

| Parent | Tiles | Current buffers | Reference buffers | Current assembly | Reference assembly | Native-window assembly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256 | 16 | 2.64 MiB | 2 MiB | 0.009 s | 0.012 s | 0.011 s |
| 512x512 | 36 | 5.64 MiB | 8 MiB | 0.041 s | 0.049 s | 0.036 s |
| 1024x1024 | 144 | 21.39 MiB | 32 MiB | 0.535 s | 0.409 s | 0.382 s |

Native layout arithmetic took roughly 0.001 seconds at each size, separate from
assembly. Both reference candidates share the same accumulator. It uses float64
sum/weight buffers; the current merger uses float32 data/weight buffers here and
allocates the padded extent. These dtype choices differ, so the table does not
establish a general memory advantage. Counts exclude input rasters, raw outputs,
GeoDataFrame overhead, temporary/final arrays, and peak RSS.

All candidates allocate full-parent buffers; none provides bounded spatial
chunking. Parent-at-a-time draining bounds the number of active parents, not the
size of a parent. A later out-of-core design needs its own evidence before
large-scene claims.

## Recommendation and next gate

Retain **Tiler for cutting/windows**, and replace live-cut result routing with
**reference-driven accumulation**. This meets validity and completion semantics
directly while keeping the existing lazy cutting behavior. Native arithmetic
does not currently justify replacing that broader behavior.

The [migration plan](../plans/2026-10-04-reference-merger.md) specifies the
replacement and atomic consumer cleanup. Its execution follows review; the
prototype is not imported by production code. Detection association and
classification aggregation remain separate from raster accumulation.

The baseline probe now calls native Tiler/Merger directly to keep this historical
comparison runnable after removal of the live-cut GeoSave API. Production
assembly uses `TileMerger(reference, parents, ...)`.

## Superseding refinement, 2026-10-05

The earlier comparison did not test using native Tiler accumulation for both
masked predictions and a coverage channel. That composition preserves valid
overlap and NaN holes without GeoSave maintaining a separate numeric accumulator.
Production now uses it: native `Merger.add` accumulates the channels, native
`Merger.merge(normalize_by_weights=False)` returns sums, and GeoSave divides by
coverage before restoring xarray coordinates. The small invalid-overlap probe
returned 8 from the valid neighbour, while default native normalization returned 0.

Tiler remains responsible for layout, padding, taper generation and accumulation.
The original overlap is explicit; no reference-spacing inference remains. This
requires a regular layout per parent and uses padded native buffers plus one
coverage channel. The historical prototype and measurements above remain evidence
of the earlier implementations, not performance claims about the refined path.

The original reference prototypes used periodic Hann. Pointwise-identical tiles
could not expose that difference. Production migration now uses symmetric SciPy
windows to preserve established Tiler weights; disagreeing-tile regression tests
compare every declared window to native Tiler/Merger.
