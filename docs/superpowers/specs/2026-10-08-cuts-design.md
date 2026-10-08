# Cuts Design: Frames and Chips as Tables of Windows

**Status:** Implemented 2026-10-08; plan in `plans/2026-10-08-cuts.md`.
Builds on `2026-10-08-stac-pystac-classes-design.md`.

**Goal:** Cut saved samples into the windows a model reads, as a table: one
row per model input, built from the item table alone, loadable as xarray,
and carried through inference so a prediction pairs back to its row and
merges back into a raster.

## Rules

1. **A cut is a table step.** Windows in, windows out, no file opened. It is
   not a raster transform; `transform/` keeps only raster-in, raster-out.
2. **The item table is the input.** A raster in memory is saved and listed as
   Items first. Facts are read through PySTAC classes via `table.to_items`.
3. **The window table is a plain GeoDataFrame.** It is derived from the item
   table and the model spec, model-specific and temporary, so it is not STAC
   and needs no extension. Persist it with `gs.to_geoparquet` if ever needed.
4. **A sample is the rows sharing `geosave:stack`,** whose groups share one
   grid. Files do not matter; concatenate tables first. Unprepared layers go
   through preprocessing, which gives them the key.
5. **Layout is rebuilt, never stored.** A chip layout is a pure function of a
   window's shape and the chip settings; cutting and merging call the same
   function.
6. **Chips are saved, then merged.** Prediction writes chip outputs keyed by
   window id; a separate step merges them.

## Flow

```python
items = stac.table.read("data/train/items.parquet")       # durable, model-independent
windows = cuts.stacks(items)                               # one whole window per sample
windows = cuts.frames(windows, 4, tolerance="10D")         # split in time
windows = cuts.chips(windows, 224, overlap=32)             # split in space
chip = cuts.select(read_stack("data/train/s0"), windows.iloc[0])   # lazy DataTree
```

`ModelSpec` declares the cuts; `FramesSpec.cut(windows)` and
`ChipsSpec.cut(windows)` pass their settings to these functions.

## Layout

```text
geodata/
├── cuts/                          NEW
│   ├── stacks.py                  stacks(items) -> windows
│   ├── frames.py                  frames(windows, length, tolerance, stride, mode) -> windows
│   ├── chips.py                   chips(windows, size, overlap, mode) -> windows; layout(shape, size, ...)
│   ├── read.py                    select, select_times, select_pixels
│   └── merge.py                   merge(windows, chips, ...) -> raster      (phase 3)
├── stac/extensions/datacube.py    NEW  write(asset, raster): time labels -> DatacubeExtension
└── transform/
    ├── chip.py                    keeps crop; chip_windows, crop_record go
    └── time.py                    DELETED; date matching moves to cuts/frames.py
```

## The window table

One row is one window on one sample. Every cut keeps these columns and
narrows them:

| Column | Meaning |
| --- | --- |
| `id` | `s0`, then `s0/frame-1`, then `s0/frame-1/chip-17` |
| `parent` | the window this one was cut from |
| `stack` | the sample; joins to the item table's `geosave:stack` |
| `times` | `{group: [labels]}`, None for a timeless group |
| `start_datetime`, `end_datetime` | what those labels span |
| `crs`, `transform` | the window's own grid |
| `row_off`, `col_off`, `height`, `width` | its pixels on the sample's grid |
| `chip` | its number in its parent's layout (chips only) |
| `halo`, `overlap`, `mode` | the layout: how a read past the parent's edge is filled, and what a merge rebuilds (chips only) |
| `geometry` | its footprint in WGS84 |

`times` is per group and explicit because a frame takes each group's own
nearest dates within a tolerance; a span alone does not reproduce that.

## What the cuts read from the item table

- **Grid:** `ProjectionExtension.ext(asset)`. Groups of a sample on different
  grids raise.
- **Time labels:** `DatacubeExtension.ext(asset).dimensions["time"].values`.
  The new Datacube module writes them for every raster carrying a time
  coordinate, a single-date scene included, so one rule covers COG scenes and
  Zarr cubes and a timeless raster is simply one that states none.

## Prediction and merging

```text
predict   trainer.predict + a BasePredictionWriter callback   -> chip outputs keyed by window id
merge     windows + chip outputs + spec.chips + items         -> one georeferenced raster per parent
```

`cuts.layout(shape, size, overlap, mode)` returns the Tiler and halo both
sides use. `ml.segmentation.callbacks.ChipWriter` appends each batch to one
Zarr store per process; `ChipWriter.read` opens them as one array along `id`,
and `cuts.merge(parents, chips, outputs, taper=...)` returns one Dataset per
parent on its grid. Validation inside the training loop keeps merging live for now; it
reads the same window table.

## Smoke-tested 2026-10-08

- Datacube time labels written through PySTAC's typed setters, sent through
  the table, and read back through the class.
- On one sample with two dated groups on their own dates, a timeless DEM and
  a label, saved as Zarr and as COG: `stacks -> frames -> chips` from the
  table alone gave the same windows, grids, footprints, per-group instants,
  pixel values and geoboxes as the existing `stack_frames` + `chip_windows` +
  `crop_record`.
- A Tiler rebuilt later from the same numbers merges chips added in any
  order back to the source: exact with no window, within 2e-7 with Hann, when
  the halo is used on both sides.
- A window table written with `gs.to_geoparquet` reads back with its `times`
  mappings.

## Breaking changes

- `transform.time` (`frames`, `stack_frames`) and `transform.chip.chip_windows`
  / `crop_record` are removed; use `cuts`.
- `FramesSpec.cut` takes and returns windows, not DataTrees.
- Model context callables read `row["times"]`, `row["crs"]`, `row["transform"]`
  in place of `raster_metadata` and `proj:*`.
- Every dated asset now declares the Datacube schema.
- Chips that overlap always sit over a halo, and a constant fill is always the
  missing value; a caller-chosen fill value is gone.
- Window ids end in `/chip-<n>` (was `/tile-<n>`); the dataset's reference
  states `stack`, `parent` and `chip` in place of `source_id`, `parent_id` and
  `tile_id`; `ChipsSpec.tiler` is `ChipsSpec.layout`.

## Out of scope

- A review of the remaining `transform/` modules.
- The `prepare_dense_data` flow.
- Annotating or filtering chips; changing validation to save-then-merge.
