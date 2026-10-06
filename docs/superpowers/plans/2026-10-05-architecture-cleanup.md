# Architecture Cleanup Implementation Plan

> Proposed cleanup order, grounded in the live checkout and smoke probes.
> Implemented. The findings below preserve the original audit baseline;
> the execution notes at the end record the selected policies and verification.

**Goal:** Make the existing modules agree about saved assets, prepared pixels,
metadata, resource ownership, and training policy after the structural refactors.

**Architecture:** Keep native xarray, GeoDataFrame/Series, Tiler/Merger, Kornia,
Lightning, and Prefect. Fix responsibilities where they currently live; remove
redundant work and speculative paths instead of adding another framework.

**Spec:** `AGENTS.md`, `docs/guides/architecture.md`, and the accepted storage
contract in `docs/superpowers/specs/2026-10-05-native-sample-inputs-design.md`.

**Tech stack:** Python 3.12+, xarray >= 2026.4.0, GeoPandas, Dask, Rasterio,
Zarr/NetCDF, PyTorch Lightning, Kornia, Prefect, HoloViews.

## Start with the actual caller flow

```python
data = read_stack("input/")
# Work on native xarray data, then save its actual result.
saved = data.gs.to_zarr(
    "dataset/sample.zarr", catalog="dataset/catalog.parquet", id="sample"
)
row = read_vector("dataset/catalog.parquet").iloc[0]
sample = row.gs.to_xarray()

# A runtime tile reference addresses an explicit prepared parent.
row = dataset.reference.iloc[0]
tile = row.gs.crop(dataset.parents[row.parent_id])
```

The first path opens stored assets. The second addresses prepared pixels already
in memory. Both operate on native objects. A reference must not claim that raw
asset pointers store its prepared pixels.

## Confirmed findings

| ID | Priority | Module and evidence | Consequence |
| --- | --- | --- | --- |
| C1 | P1 | `ml/segmentation/supervised/data.py:114–119` copies every missing source column, including `assets`, into prepared references. | The reference describes `image`/`label`, but reopening its assets returns raw `optical`/`label`. A doubling preprocess produced different pixels through the same row. |
| C2 | P1 | `geodata/io/__init__.py:205` and `io/layout.py:258–289` build stacks/cubes without keeping opened readers' close callbacks. | Closing a directory stack or reconstructed COG cube does not close the readers it opened. The newer asset reader preserves callbacks, so entry points disagree. |
| C3 | P1 | `geodata/core/stack.py:292` uses default `to_dataset()` inheritance; `io/zarr.py` already uses `inherit="all_coords"` for group reads. | A native tree with root scalar `time` and `spatial_ref` returns groups lacking both coordinates through `.gs.rasters`. Pixel grids may still resolve, but metadata is incomplete. |
| C4 | P1 | `io/zarr.py:127,213–217` stores caller-spelled source paths; GDAL resolves local paths. `io/netcdf.py:126–135` does not record each group's source/selector. | Registration fails after changing cwd for a relative Zarr source. A freshly reopened multi-group NetCDF cannot be registered by `GeoVector.from_xarray`. |
| C5 | P1 | `workflow/tasks/labels.py:65–73` checks traversal and absolutes but not empty/root or canonical names. | `""` and `"."` address the output root; `"a/./b"` and `"a/b"` are distinct IDs addressing the same directory. |
| C6 | P1, policy | `ml/segmentation/supervised/module.py:258,281–289` expects every reference tile locally; DataModule uses ordinary tile DataLoaders. | Native distributed sampling can split one scene across ranks. In a two-tile/two-rank probe, neither rank completes or scores a scene. |
| C7 | P2 | `model/encoder/clay.py:165` reads `MODEL_SOURCE` before choosing the explicit checkpoint path. | A supported smaller architecture with a local checkpoint raises `KeyError` when it has no published Hub entry. |
| C8 | P2 | `core/vector.py:644–661` opens assets for validation, closes them, then `from_assets` opens them again. `io/assets.py:100–102` opens saved data before invoking this sequence. | Companion catalog creation repeats metadata discovery instead of constructing its record from one authoritative opened snapshot. |
| C9 | P2, ownership | `ml/transforms/augmenter.py` supports custom YOLO and several unwritten training methods; only supervised segmentation consumes it. | Shared code owns hypothetical policies despite the rule to share across head types only after two written modules repeat behavior. |
| C10 | P2 | `io/netcdf.py:205–208` returns native xarray's delayed result; Zarr wraps completion to return its saved Path. | Deferred NetCDF computes to `None`; callers cannot rely on the same saved-path return contract across formats. |

These findings are about current behavior. The mixed staged/unstaged checkout
does not establish which historical refactor introduced each one.

## Global constraints

- Preserve the dirty checkout and unrelated edits; no compatibility aliases.
- Keep `row.gs.to_xarray()` as saved-asset opening with an optional stored window,
  and `row.gs.crop(parent)` as explicit window application to native prepared data.
- Keep neural models tensor in and raw output out; do not add encoder-owned
  geodata loading or model preparation methods.
- Keep metadata registration and pixel-window reads lazy.
- Preserve CRS, transform, coordinates, time, nodata, dtype and variable identity.
- Keep native Tiler/Merger; geometry does not reconstruct their pixel layouts.
- Keep sample staging/publication policy in the existing workflow module.
  Companion writers and ingest catalog upsert intentionally have different policies.
- Keep shared-grid/time registration restrictions and non-atomic companion
  publication explicit; this cleanup does not add independent-grid catalogs.
- No new dependencies, generic job/runner classes, model downloads, wheels,
  publishing, notebook edits, or automatic generated-workspace rewrites.

## Task 1: Consistent native reads and file ownership

**Findings:** C2, C3, C4. This is the first implementation slice.

**Files:** `geodata/core/stack.py`; `geodata/io/{__init__,assets,layout,storage,
zarr,netcdf,gdal}.py`; tests in `tests/geodata/core/test_stack.py` and
`tests/geodata/io/{test_assets,test_read_stack,test_layout,test_zarr_stack,
test_netcdf,test_storage}.py`.

- [x] Add failing behavior tests for directory/COG readers closing all files,
  including an exception while opening a later asset or combining leaves.
- [x] Add a native DataTree test with scalar root time/CRS coordinates. Extracted
  groups must retain them through `.gs.rasters`, transformations, and persistence.
- [x] Add relative-path/cwd-change registration tests and multi-group NetCDF
  href/group round trips, alongside existing Zarr tests.
- [x] Use native coordinate inheritance consistently. Apply existing filesystem
  resolution in one location instead of several `"://"` checks. Preserve source
  href/group as one pair for both supported grouped formats.
- [x] Give each I/O opener ownership of the readers it opens. Keep that ownership
  through returned native trees/cubes and close completed reads on failure.
  Document the pure `stack()` factory's ownership; do not silently turn every
  borrowed Dataset into an independently owned reader.
- [x] Remove repeated close/path handling only after these behaviors are covered.

**Result:** Opening through a row, directory, COG tree, Zarr, or NetCDF follows
the same metadata and lifecycle rules. Locality improves in geodata I/O without
changing the native API.

## Task 2: One registration pass and consistent writer results

**Findings:** C8, C10.

**Files:** `geodata/core/{vector,raster,stack}.py`; `geodata/io/{assets,netcdf}.py`;
`geodata/stac/item.py`; tests in `tests/geodata/core/test_catalog.py`,
`test_vector.py`, and `tests/geodata/io/test_netcdf.py`.

- [x] Add an integration assertion that each registered asset is opened once
  for a registration operation, without running pixel tasks. Keep actual saved
  metadata authoritative and preserve selection/grid/shape rejection behavior.
- [x] Have registration construct and validate its record against the same opened
  snapshot, using the existing STAC metadata module. Do not solve repeated I/O
  with a second mutable catalog class or a global reader cache.
- [x] Make deferred NetCDF completion return its Path, matching Zarr; expose
  `compute` consistently in the relevant accessor signatures and docstrings.
- [x] Retain changed-pixel, moved-directory, overwrite, failure-publication,
  scalar time and group-pointer round-trip tests.

**Result:** Registration constructs metadata in one pass;
writers agree on saved-path results. Do not add NetCDF companion catalogs as an
unrequested feature while fixing its existing write contract.

## Task 3: Honest prepared references

**Finding:** C1.

**Files:** `ml/segmentation/supervised/data.py`; `tests/ml/segmentation/supervised/
test_data.py`; active architecture and segmentation workspace guidance.

- [x] Add a regression using preprocessing that renames a raster and changes
  its values, then one that selects a temporal frame. Source pointers must not
  be presented as assets storing the prepared result.
- [x] Replace blanket copying of source columns with explicit retention of
  annotations and source identity. Keep source asset descriptions as provenance,
  separate from active prepared asset pointers; do not change IDs by parsing them.
- [x] Keep runtime references usable through `row.gs.crop(parent)`. A reference
  may advertise readable prepared assets only after those parents have been saved
  and registered. Do not put preprocessing inside the geodata row reader.
- [x] Preserve metadata-only setup, stable frame/tile IDs, reordered references,
  multiprocessing reopening, original targets, validity and whole-scene scoring.

**Result:** Reference metadata and asset pointers have one clear meaning. The
training method owns construction of its prepared parents and batches; geodata
continues to open exactly the assets a row names.

## Task 4: Narrow workflow and model construction repairs

**Findings:** C5, C7. Implement these as separate small changes.

**Files:** `workflow/tasks/labels.py`, `tests/workflow/tasks/test_labels.py`,
`tests/workflow/flows/test_prepare_dense_data.py`; `model/encoder/clay.py`,
new `tests/model/encoder/test_clay.py`.

- [x] Reject empty, root and noncanonical directory IDs in the label-table
  reader before task submission. Check actual destination uniqueness and retain
  valid nested IDs. Do not normalize or constrain IDs on generic GeoVectors.
- [x] Assert existing scene directories/manifests remain untouched for invalid IDs.
- [x] Add a smaller Clay architecture/local-checkpoint test with mocked weight
  loading and a Hub-call assertion. Consult the published source map only when
  downloading; preserve checkpoint validation and no-pretraining behavior.

**Result:** The label module owns its filesystem-ID invariant and the encoder
builder owns its checkpoint choice, each with a focused behavior test.

## Task 5: Method-owned augmentation and scene evaluation

**Findings:** C6, C9. These are policy changes rather than mechanical file moves.

**Files:** `ml/transforms/{__init__,augmenter}.py`, `ml/segmentation/supervised/
{data,module}.py`, their mirrored tests and segmentation workspace guidance.

- [x] Preserve existing joint image/target transforms, shared temporal geometry,
  nodata/ignore handling, empty pipelines and per-band normalization tests.
- [x] Retain `ImageAugmenter` as the YAML configuration entry point, subclassing
  native `AugmentationSequential`; the user explicitly requested preserving it
  during execution. Remove unused YOLO conversion and redundant forwarding.
  The DataModule owns when and how augmentation runs. Keep logits interpretation
  separate from augmentation.
- [x] Select the plan's interim policy: reject multi-device validation/test
  explicitly before evaluation begins. Verify validation/test hooks and an
  actual two-process CPU Lightning execution report the unsupported mode.
- [x] Preserve single-device scene completion, original categorical labels,
  validity-weighted logits, and intentional incomplete-run exclusion.
- Complete-parent rank assignment, uneven/empty rank completion, and distributed
  metric synchronization remain a separate training policy design. No general
  distributed merger framework was introduced.


**Result:** The Module/DataModule pair owns the sampling and completion policy
its metrics require. Shared code reflects written consumers, not future heads.

## Task 6: Active docs, public types and tooling

**Files:** `docs/guides/architecture.md`; segmentation workspace README;
`src/geosave_engine/model/README.md`; `.pre-commit-config.yaml`; the source files
reported by the whole-package BasedPyright baseline below.

- [x] Remove active `data=` examples at architecture guide lines 208–209 and
  starter README lines 113–114. Correct the starter's query/load ownership and
  claim that Prithvi/Clay context is already enabled. Update copied template
  source and test a newly generated workspace.
- [x] Keep superseded design documents marked historical; do not rewrite every
  archived proposal or edit the existing consumer workspace as library source.
- [x] Inspect and correct public types for native GeoDataFrames, raster variable
  keys, HoloViews layouts/maps, metadata Scope, encoder pyramids, and third-party
  call signatures. Separate genuine contract errors from inaccurate dependency
  annotations; do not change valid array behavior merely to suppress diagnostics.
- [x] Align the pre-commit Ruff pin with the locked Ruff version and remove dead
  selectors for the former template paths. Keep notebook contents outside scope.
- [x] Run whole-source checks so passing checks for edited files do not conceal
  unresolved diagnostics elsewhere in the same public module flow.

**Result:** Users see the API the implementation actually provides, with
the same paths and conventions in source, generated starters and documentation.

## Verification and review focus

Run focused tests after each change; run the default suite once after integration:

```bash
uv run pytest
uv run ruff check src tests
uv run basedpyright src/geosave_engine
uv run zensical build
git diff --check
```

Run the explicit slow worker/import checks when changing readers or DataModules;
distributed evaluation also needs a local two-process Lightning check. No wheel
build is needed for these changes.

Review focus: partial opens close earlier readers; root scalar time/CRS survives
group extraction; source pointers survive cwd changes; raw and prepared metadata
cannot be mixed under one active asset field; path-equivalent IDs cannot publish
to one destination; rank partitioning must preserve complete-scene evaluation.

## Audit checks and limits

- Focused existing regression: **79 passed**, one upstream deprecation warning.
  Passing tests do not exercise all the reproduced gaps above.
- Ruff: **passed** for source/tests.
- Whole-source BasedPyright: **24 errors**, zero warnings. Areas: cloud-mask
  SciPy calls (3), GDAL keys (1), GeoJSON/GeoPackage (4), GeoParquet returns (2),
  GeoTIFF import typing (1), geolocator (3), geometry mapping (1), visualization
  return/argument types (5), encoder pyramids (3), raster requirement Scope (1).
- Smoke probes reproduced C1–C7 and C10, using temporary local files or mocked
  backends; C8/C9 follow directly from current call sites and consumer searches.
- Probes showed group extraction drops scalar `time` and `spatial_ref` while the
  grid may still resolve. Do not equate missing CF coordinates with absent pixels.
- The distributed probe partitions scene tiles with native `DistributedSampler`;
  it is not a full multi-process Lightning execution.
- These counts describe the original audit, before implementation.

Defer new model-input abstractions, automatic context recomputation after spatial
augmentation, independent-grid catalogs, regression/detection training methods,
and the Panel explorer. They need their own concrete caller-flow design after
the storage/reference contracts above are consistent.


## Execution notes

All six cleanup slices are implemented. Reader lifecycle and inherited
coordinates are consistent, source/group pointers survive cwd changes, catalog
registration uses one opened snapshot, and deferred NetCDF returns its path.
Prepared references retain raw provenance under `source_id`/`source_assets`
instead of claiming those paths contain their prepared pixels. Label IDs and
local Clay checkpoint selection have focused fixes.

The user requested retaining `ImageAugmenter` during implementation. Its public
import and YAML `name`/`init_args` configuration remain; it now subclasses native
Kornia `AugmentationSequential`, including nested configurations, default crop
sizes, and empty-pipeline identity. Unused custom YOLO conversion helpers and the
redundant forward wrapper were removed.

Scene validation/test deliberately require a single device. Distributed scene
partitioning remains deferred under the plan's explicit rejection alternative.
The final read-only review found a companion Dataset metadata naming mismatch;
a round-trip regression reproduced it, and metadata now uses actual asset names
without another read. There were no other review findings.

Work stayed in the shared dirty checkout to preserve the prior refactor; no
commit or publication was performed. Unrelated changes were outside the cleanup
review. Existing shared-grid/time registration and non-atomic companion
publication constraints remain.

Final verification: **1,413 default tests passed**, **7 explicit worker/import
and two-process tests passed**, Ruff passed, whole-source BasedPyright reported
**zero errors**, documentation build passed, and `git diff --check` passed.

### Augmentation correction (2026-10-06)

The user requested restoring the explicit supported `DataKey` list and YOLO
box conversion. This supersedes the YOLO removal in task 5: these formats are
part of the configurable training augmentation API even before a built-in
detection training method exists. `ImageAugmenter` contains one native Kornia
pipeline; it converts normalized YOLO boxes to pixel `xyxy` at its boundary
and normalizes results with the output image size. Pascal VOC, COCO, masks,
keypoints, and class labels continue through Kornia's native handlers.

The follow-up API correction moves `data_keys` to the augmentation call. One
image defaults to `input`; multiple tensors require explicit keys. The wrapper
is an `nn.Module` containing Kornia rather than subclassing its container.
The unused `inverse()` and parameter-replay interfaces were removed, and the
supervised DataModule passes its joint image/mask keys explicitly at each call.
