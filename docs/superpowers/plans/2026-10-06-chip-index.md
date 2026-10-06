# Persisted chip index implementation plan

> Use superpowers:executing-plans inline. The user has authorized implementation
> in the current shared checkout; preserve existing changes and leave it uncommitted.

**Status:** Writer path refinement and chip/Tiler vocabulary changes are complete.
Asset/Item construction and chip indexing remain pending their metadata and
selection design. The proposed `sources` dictionary API was withdrawn.

**Goal:** Explicitly generate reusable chip GeoParquet, then train from its rows.

## Writer return refinement — approved implementation

The user authorized implementation of the path contract on 2026-10-06. IO returns
locations, not xarray objects paired with locations. A path supplies an asset's
href; file/store metadata supplies its actual contents. Paths need not encode band
identity or timestamps, and a directory scan is not the record of a write's outputs.

Return convention:

- One physical file or store: `Path`.
- A raster export capable of producing multiple COGs: `list[Path]`, even for one
  output. The list contains exactly the files successfully produced by that call.
- GeoStack COG export: `dict[str, list[Path]]`, retaining the existing group names.
- Deferred store writes resolve to the same `Path` after they complete.
- DataArray COG writes remain single-file operations returning `Path`.

The existing `io.geotiff.write_cog` returns one `Path`. The old
`io.layout.write_leaves` has been consolidated into `write_tree`, which returns
the actual `list[Path]`. Public raster/stack exporters propagate those inventories
instead of returning directories. No result wrapper or compatibility alias.

Catalog construction remains separate. It consumes output locations and native
metadata, builds complete PySTAC Assets and Items, then delegates table conversion
and serialization to stac-geoparquet. Local paths become href strings with an
explicit base or absolute location; existing URLs keep their URI syntax. Do not
change relative hrefs after insertion into the GeoDataFrame.

COG band count, descriptions/order, dtype, nodata and grid come from file headers.
Scene time comes from written metadata or explicit source metadata, rather than
requiring a filename convention. NetCDF/Zarr use native xarray/CF variables,
groups and time coordinates; the source Item can describe their full time span.

A flat list of paths alone cannot identify original GeoStack groups when
variables have the same names. The stack writer retains group names in a native
mapping; standalone raster writers do not require a GeoStack.
Paths do not establish imagery-versus-label purpose or scene membership by themselves.

Disposable smoke on 2026-10-06: two dates, two bands, joined and split COG exports;
arbitrary physical filenames; native PySTAC EO/Raster/Projection metadata;
STAC GeoParquet round trip; lazy `odc.stac.load`; identical band names, timestamps
and pixels. Exit status zero. Stable asset names across scene Items were required
for the tested ODC band mapping; using different asset names per date produced
missing bands. Keep identity independent of physical filenames.

Library evaluation: rio-stac provides native Item construction from raster files,
but its current source names EO bands `b1`, `b2` (descriptions retain file band
descriptions), and raster metadata extraction reads sampled statistics. It is
not installed or adopted. Do not assume it satisfies our identity/laziness contract.

### Writer execution tasks

- [x] Test exact output inventories for timeless, scalar-time and multi-date
  rasters; joined/split bands; explicit filenames; pre-existing unrelated files.
- [x] Consolidate `write_tree` and `write_leaves` into one physical tree writer
  returning `list[Path]`; propagate through the raster accessor. Preserve dotted
  variable and group stems rather than truncating them as suffixes.
- [x] Return a native group-to-paths mapping from GeoStack; round-trip groups
  sharing variable names and distinct pixel values. Migrate directory reader
  callers to use their explicit directory input.
- [x] Document the changed returns; verify format/store IO, native STAC loading
  from actual file headers and stac-geoparquet, Ruff, types and diff checks.
- [x] One fresh final review of this writer delta against the baseline snapshot.

Asset/Item builders and indexed chip readers remain pending their separate
metadata and selection design. This step does not introduce new STAC inference.

Writer review focus: returned paths must name every file written and exclude
unrelated files. Dotted stems, explicit TIFF suffixes, scalar times, non-unique
band names across groups, and native write options must survive the boundary.

**Architecture:** Native source Items describe saved pixels. TIFF Items describe
dated scenes; Zarr/NetCDF Items can describe whole time spans. Native Tiler creates
spatial windows on a chosen tile and temporal frame. A chip table retains those
windows and source Items; the Dataset consumes this table rather than retiling.

**Spec:** The accepted spatial hierarchy is scene/store -> tile -> chip; frames
are temporal. Assets describe physical files independently of a GeoStack's
logical groups. Semantic labels are aligned class-ID rasters in a GeoStack.

**Constraints:** No new dependency, STAC schema reconstruction, compatibility
aliases, generic inference of layer names, implicit reprojection, or pixel writes.
Keep native Tiler/Merger. Source JSON uses PySTAC serialization, not a GeoSave
metadata codec. Asset and Item builders remain the separate pending API.

## Task 1: Chip generation and saved source reading

- [ ] Add `geodata.chip(data, *, id, source, tiler, padding, properties)` returning
  a native GeoDataFrame. Persist tile grid/shape, native Tiler parameters, native
  chip index, exact pixel window, ordered temporal selection and source Items.
  This API is provisional until writer output and asset selection are settled.
- [ ] Replace `from_layouts` with this single operation and migrate callers.
- [ ] Implement row reads for chip sources and native catalog assets, using
  existing format readers and xarray combination. Select original timestamps;
  use ODC's aligned-grid slicing and existing lazy padding.
- [ ] Prove metadata construction stays lazy, TIFF split bands and multiple dates
  match whole-store NetCDF/Zarr selection, and concatenated Parquet chips retain
  identity and reassemble using a restored native Tiler after shuffled reads.

## Task 2: Dataset consumes the saved index

- [ ] Accept a Parquet path or native chip frame; one row is one Dataset sample.
  Apply model raster selection and value preprocessing after reading a chip;
  changing the grid must happen before indexing.
- [ ] Keep process-owned lazy readers and parent reads only for evaluation.
  Restore native Tilers from table metadata; preserve chip ID through batching.
- [x] Rename `TilesSpec`/`spec.tiles` to `ChipsSpec`/`spec.chips`, `layout()` to
  `tiler()`, and consumer `layouts` to `tilers`; update templates and tests.
- [ ] Replace obsolete loader tests with persisted-index fixtures and remove
  reader xfails. Verify batching, time windows, context and native merging.

## Task 3: Documentation and verification

- [ ] Document explicit frame/chip generation, ordinary GeoParquet concatenation,
  semantic labels and TIFF-versus-store source granularity. Classification and
  detection can share chip selection and identity; their target/merge policies
  remain task-owned.
- [ ] Run focused raster/row/chip/Parquet, model-spec and segmentation suites,
  Ruff, scoped BasedPyright and diff checks. One fresh final review of this delta.

## Review focus

Filtered chip tables must not claim full-tile evaluation coverage. Fringe and
halo padding must retain the native Tiler index. Source Items must remain unchanged.
Mismatched grids must fail rather than resample. Temporal order must survive storage.
