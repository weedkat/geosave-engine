# Training Data Path Implementation Plan

> **For agentic workers:** executed inline with superpowers:executing-plans and
> superpowers:test-driven-development. Checkboxes are the progress ledger.

**Goal:** The segmentation template trains end to end from prepared samples
listed in a STAC manifest.

**Architecture:** A manifest is a STAC table held by `GeoVector`. The model spec
declares the cuts (`frames`, `tiles`), the model `inputs`, and the tensor
`transforms`. `supervised.Dataset` turns manifest rows into tiles;
`supervised.DataModule` augments and transforms batches on the device;
`supervised.Module` scores whole rasters in validation.

**Spec:** `docs/superpowers/specs/2026-10-03-training-data-path-design.md`

## Global Constraints

- Do not commit. Work on `main` over the uncommitted tree.
- Never `git checkout --`, `restore`, `stash`, or `clean`. No regex renames
  across files; edit file by file.
- `geodata` imports no torch; `model.spec` and `workflow` load no torch.
- No compatibility aliases for renamed or removed names.
- Every behaviour change starts with a failing test. `uv run pytest` is green
  after every task.

## Review Focus

1. A raster opened in the parent process and read in a forked DataLoader worker.
2. A manifest row that lacks a layer the spec's `inputs` or `target` names.
3. A validation run cut short (sanity check): no raster completes.
4. A `(B, T, C, H, W)` batch through augmentation and transforms.
5. A manifest moved to another folder together with its samples.

---

## Task 1: Encoders declare what their pyramid holds

The template chain `dinov3 + dpt + segmentation` does not build: encoders
annotate `forward_pyramid` as `tuple[list, list]`, decoders take
`list[torch.Tensor]`.

**Files:** `model/encoder/{dinov3,prithvi,clay}.py`, `tests/model/test_registry.py`

- [x] Test: `build_model` of `dinov3 + dpt + segmentation` returns a chain whose
      `inputs` are `{"image"}`, and of `prithvi_tl + dpt + segmentation` returns
      `{"image", "temporal_coords", "location_coords"}`. Fails with `TypeError`.
- [x] Annotate every encoder `forward_pyramid` as
      `tuple[list[torch.Tensor], list[torch.Tensor]]`.
- [x] Suite green.

## Task 2: Encoders own their context functions

**Files:** `model/encoder/prithvi.py`, `model/encoder/clay.py`,
`ml/datasets/tiles.py`, their tests.

**Produces:**

```python
# model/encoder/prithvi.py
def temporal_coords(data: xr.Dataset | xr.DataArray) -> torch.Tensor  # (T, 2) year, zero-based day of year
def location_coords(data: xr.Dataset | xr.DataArray) -> torch.Tensor  # (2,) latitude, longitude
# model/encoder/clay.py
def time(data) -> torch.Tensor     # (2,) ISO week, hour; one time label only
def latlon(data) -> torch.Tensor   # (2,) latitude, longitude
```

- [x] Tests: each function on a small raster with a known time and grid; Clay
      `time` refuses two time labels.
- [x] Replace the two `model_context` static methods with these functions.
- [x] Remove `TileDataset(model_context=)` (moved to Task 8; see Rulings).
- [x] Suite green.

## Task 3: The spec declares cuts, inputs, and transforms

**Files:** `model/spec/model.py`, new `model/spec/cuts.py`,
`model/spec/__init__.py`, `tests/model/spec/`.

**Produces:**

```python
class FramesSpec(SpecModel):
    length: PositiveInt
    stride: PositiveInt | None = None
    tolerance: str
    mode: FrameMode = "strict"
    def cut(self, sample: xr.DataTree) -> list[xr.DataTree]

class TilesSpec(SpecModel):
    size: PositiveInt | tuple[PositiveInt, PositiveInt]
    overlap: NonNegativeInt = 0
    mode: TilingMode = "reflect"
    window: StitchWindow | None = None
    @property
    def shape(self) -> tuple[int, int]
    def cut(self, rasters: Sequence[Raster]) -> Tiles

class ModelSpec:
    frames: FramesSpec | None = None
    tiles: TilesSpec | None = None
    inputs: dict[Name, Ref | CallSpec] = {}
    transforms: dict[Name, list[dict[str, Any]]] = {}
    @property
    def pixel_inputs(self) -> dict[str, str]       # model input -> raster it reads
    def model_inputs(self, tile: Mapping[str, Any]) -> dict[str, Any]
```

- [x] Tests: YAML round trip with mixed `!ref` and call entries; `model_inputs`
      resolves a reference to the raster and invokes a call; an `inputs`
      reference naming no step or raster is refused on load; a `transforms` key
      naming no pixel input is refused on load; `window` with `overlap: 0` is
      refused on load; loading still imports no torch.
- [x] Implement.
- [x] Suite green.

## Task 4: The time cut is named frames

**Files:** `geodata/transform/time.py`, `geodata/errors/{errors,__init__}.py`,
`tests/geodata/transform/{test_time,test_stack_metadata}.py`, docs that name it.

| Today | Becomes |
| --- | --- |
| `window(data, size, stride=)` | `frames(data, length, stride=)` |
| `window_stack(tree, size, tolerance=, stride=, mode=)` | `stack_frames(tree, length, tolerance=, stride=, mode=)` |
| `WindowMode` | `FrameMode` |
| `DroppedWindowsWarning` | `DroppedFramesWarning` |

- [x] Rename in tests first; they fail on import.
- [x] Rename in source, file by file. "Window" stays only where it means
      blending weights.
- [x] Suite green.

## Task 5: A reflected tile reaches past the raster

`Tiles(..., mode="reflect")` raises `conflicting sizes for dimension 'y'` on the
last tile of a 510x510 raster cut to 224.

**Files:** `geodata/transform/tiling.py`, `tests/geodata/transform/test_tiling.py`

- [x] Test: every tile of a 510x510 raster cut to 224 with overlap 32 has shape
      224x224 for each of the four modes, and a `boxcar` merge of the tiles
      rebuilds the raster exactly.
- [x] Find the cause with superpowers:systematic-debugging, then fix it.
- [x] Suite green.

## Task 6: A manifest is a STAC table

**Files:** `geodata/core/vector.py`, `geodata/utils/io/{geoparquet,__init__}.py`,
`tests/geodata/core/test_vector.py`, `tests/geodata/utils/io/`.

**Produces:**

```python
GeoVector.from_xarray(data, *, assets=None, id=None, datetime=None,
                      geometry=None, crs=None, **properties)
```

Columns: `id`, `type`, `stac_version`, `stac_extensions`, `links`, `geometry`,
`datetime`, `start_datetime`, `end_datetime`, `proj:code`, `proj:shape`,
`proj:transform`, `assets`, then caller properties. `bbox` is written by the
writer.

- [x] Tests, record: columns present; `id` defaults to the anchor stem; an asset
      key naming no layer is refused; a timeless raster is refused without
      `datetime=`; band names land in `assets.<layer>.bands`.
- [x] Tests, round trip: hrefs are stored relative to the file and read back
      openable after the folder is moved; null and repeated ids are refused; a
      projected table is refused; a row lacking an asset reads back without it;
      `stac_geoparquet` reads the file and every item validates with pystac.
- [x] Implement `from_xarray` and `from_anchor` column names.
- [x] Implement the writer and reader rules for a table with `assets`; remove
      the `path` column handling.
- [x] Suite green.

## Task 7: Dense preparation writes that manifest

**Files:** `workflow/tasks/{sample,manifest,dense}.py`,
`workflow/flows/prepare_dense_data.py`, `tests/workflow/`.

- [x] Tests: a prepared sample is a folder holding one raster per layer; the
      manifest row's assets open with `read_raster` and stack onto one grid;
      caller metadata columns are kept.
- [x] Samples are written per layer and opened with `read_raster` plus `stack`.
- [x] The manifest is built from `GeoVector.from_xarray(..., assets=...)` records.
- [x] Suite green.

## Task 8: Dataset

**Files:** `ml/segmentation/supervised/data.py`,
`tests/ml/segmentation/supervised/test_data.py`.

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, manifest: GeoVector, spec: ModelSpec, *, target: str = "label",
                 ignore_index: int = 255) -> None
    tiles: Tiles                      # re-cut in each process
    def merger(self, *, window: StitchWindow | None) -> TileMerger
    def __getitem__(self, index) -> tuple[dict[str, Tensor], Tensor, int]
```

- [x] Tests: length equals the tile count over all samples; a sample holds the
      declared inputs, a `(H, W)` long target, and its index; a row lacking the
      target layer is refused, naming the row; two DataLoader workers read a
      batch.
- [x] Implement.
- [x] Suite green.

## Task 9: DataModule

- [x] Tests: `setup("fit")` builds both datasets; the hook augments only while
      training and moves image and target together; transforms run in every
      split; NaN pixels become 0; a `(B, T, C, H, W)` input keeps its shape.
- [x] Implement.
- [x] Suite green.

## Task 10: Module

- [x] Tests: the constructor takes no `in_channels` or `input_size`;
      `validation_step` scores a raster only once all its tiles arrived, on
      exactly its own pixels.
- [x] Implement.
- [x] Suite green.

## Task 11: Template trains

- [x] Template `model_spec.yaml` gains `tiles`, `inputs`, `transforms`;
      `train.yaml` gains `spec`, `train`, `val`, `target`, `augmentations` and
      moves `in_channels` and `input_size` onto the encoder.
- [x] Example manifest rewritten in the new shape.
- [x] Slow test: one `fast_dev_run` fit on the example samples with the template
      configuration and a small encoder.
- [x] Docs: `AGENTS.md`, `README.md`, guides, template README.

## Rulings

- Task 1: Ruling: encoders declare `tuple[list[torch.Tensor], list[torch.Tensor | None]]`,
  the type the DPT decoder already takes — the chain matches types exactly —
  cost if wrong: one annotation per encoder.
- Task 2: Ruling: `TileDataset(model_context=)` stays until Task 8, where it takes
  the spec's `inputs` instead — removing it first would leave prediction with no
  way to pass encoder context — cost if wrong: one argument.
- Task 3: Ruling: `TilesSpec.size` is an int or a two-item list, not a tuple, because
  the spec's safe YAML dumper cannot write tuples. `transforms` entries are
  `TransformSpec(name, init_args)`. `ModelSpec.model_inputs` does not revalidate
  the spec per call; it runs once per tile.
- Task 5: cause: dask's `reflect` and `wrap` padding stop after one pass, so a lazy
  tile that must invent more pixels than it holds came out short. Fixed by
  letting numpy lay out the repeated positions and indexing with them.
- Task 6: Ruling: a relative href means relative to the manifest, as the existing
  `stored_asset_path` and `resolve_asset_path` already read it; an absolute href
  below the manifest's folder is shortened on write — cost if wrong: one rule in
  the writer.
- Task 6: Ruling: `datetime` is null unless the caller passes it. An anchor's
  timespan covers whole days, so start and end never coincide and no instant
  can be inferred — cost if wrong: one column.
- Task 6: Ruling: the grid CRS is `proj:code` (`EPSG:n`) when it has an EPSG
  code and `proj:wkt2` otherwise; `str(crs)` is WKT for a CRS read from a file.
- Task 6: Ruling: `from_xarray` loses `path=` and the `variables` field; band
  names live in each asset's `bands`.
- Task 7: Ruling: `write_sample` and `open_sample` stay, because they hold the
  atomic publication and the completed-sample check that `stack.gs.to_cog` does
  not. A Zarr sample is now a folder of per-layer stores, like a GeoTIFF one, so
  `sample_path` and the manifest's `format` column are gone. The item `id` is
  the workflow's sample ID — cost if wrong: prepared Zarr samples from before
  this change no longer open.
- Tiling cleanup (asked for mid-run): each raster is extended once at
  construction, so a tile is a plain selection; the unused `Tiles.grid` is gone;
  the merger keeps one record per raster instead of four parallel maps. A tile
  past the raster now continues the raster's own extension — before, the frame
  and the trailing reach were padded in two passes that disagreed.
- Task 8: Ruling: `TileDataset(tiles, spec=None)` reads tiles through the spec's
  `inputs`, replacing `model_context=`. `read_inputs(spec, tile)` is shared with
  `supervised.Dataset`.
- Task 8: Ruling: workers start with `forkserver`. After a fork, a child's exit
  breaks the parent's raster handles, and a parent that has read pixels makes a
  forked child hang. The dataset also releases what it opened to count — cost
  if wrong: worker start-up is slower than a fork, by the time to import torch.
- Task 9: Ruling: `transforms` is keyed by input name; `ImageAugmenter` keeps its
  name; an invented target pixel becomes `ignore_index`; NaN pixels become 0
  after the transforms.
- Task 10: Ruling: the module reads its mergers from the loader's dataset
  through the trainer, and warns when a cut-short validation scores nothing.
- Task 11: Ruling: `examples/data/dw_imagery/manifest.parquet` was not rewritten.
  It is a data file with uncommitted changes; the end-to-end test builds its own
  manifests from the example sample folders.
- Final review: self-review (subagents are not permitted in this session).

