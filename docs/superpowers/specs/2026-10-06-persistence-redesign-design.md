# Raster persistence and STAC interoperability

Status: direction agreed; representation and dependency selection require the
evidence checkpoint below. This document supersedes the deferred stack and row
API decisions in `2026-10-06-cog-leaves-catalog-design.md`. It does not authorize
product implementation by itself.

## Agreed outcome

```python
# Native STAC ingestion uses the library's conversion, not GeoSave Item assembly.
records = gpd.GeoDataFrame.from_arrow(
    parse_stac_items_to_arrow([item], drop_invalid_properties=False).read_all()
)
catalog = catalog.gs.upsert(records, on="id")
row = catalog.set_index("id").loc[item.id]

sample = row.gs.to_stack()                  # xr.DataTree
raster = row.gs.to_raster(layer="optical")   # xr.Dataset
array = raster.gs.to_array()                # xr.DataArray; existing conversion

catalog = read_vector("catalog.parquet")
raster = catalog.gs.to_raster("forest")     # existing grouped Dataset reader
```

Saved catalogs must support both:

1. Exact reconstruction of supported persisted GeoSave rasters and stacks.
2. Valid STAC records with assets that other clients can identify and open.

Keep native GeoDataFrames for selection, spatial queries, annotations, and
editing. Do not create a catalog wrapper or a parallel xarray persistence model.
Replace `GeoRow.to_xarray` with explicit readers, without an alias.

## Simplification is the primary design criterion

Adopt the common STAC Item/Asset convention, integrate GeoSave operations with
it, and delete the alternate structural representations where possible.
Native `pystac.Item` objects from other producers must be usable without adding
`raster_metadata`, `geosave:raster`, or `geosave:format` just to open their assets.
The library conversion above already supplies a native GeoDataFrame; do not
invent an insertion framework or duplicate conversion merely to shorten it.

Asset keys are the default selectors for ordinary STAC Items. Use PySTAC's own
href resolution for relative assets, with an Item/document location supplied
when necessary. Respect asset media types, roles, and standard opening options.
Do not assume every asset is raster data, or every Item's rasters share a grid.
Selection and supported native readers determine what can be opened.

GeoSave-specific exact reconstruction is an additional contract for supported
GeoSave writes, not a prerequisite imposed on third-party STAC Items. If a
common convention cannot restore an arbitrary original xarray structure, use a
native format that can, or narrow the supported COG contract. Do not compensate
by growing a universal reconstruction schema.

Every retained GeoSave branch must own an actual domain invariant that a native
library does not own. Do not maintain STAC core/extension field whitelists,
version/schema metadata, or parallel Asset structures. Use available PySTAC
extension and media-type APIs rather than duplicate constants. Tests must
exercise externally constructed Items, native files, and observable behavior;
they must not dictate proprietary keys or implementation-specific routing.

The implementation review must list deleted responsibilities and justify
remaining custom metadata with its concrete consumer and missing ecosystem
capability. A new typed model is not itself simplification.

## Scope and constraints

- Preserve unrelated working-tree changes; do not commit existing user edits.
- Ignore notebooks.
- Dependencies point one way: `geodata <- model <- ml`.
- Raster persistence remains GeoTIFF/COG, Zarr, and NetCDF.
- Prefer existing dependencies; adding or replacing a library requires smoke
  evidence against the same fixtures, not an API demonstration alone.
- Opening, catalog construction, and reconstruction remain metadata-only and
  lazy; no raster pixel tasks run until requested.
- Preserve supported grids, CRS, coordinates, variable identity and order,
  dtypes, nodata, attrs, exact timestamps and time-axis shape, and layer identity.
- No implicit reprojection, resampling, dtype conversion, missing-band filling,
  or spatial alignment.
- Ordinary tile references remain GeoParquet. An assets column alone does not
  make a reference STAC; undated and pixel-only references remain supported.
- No compatibility aliases or duplicate publication paths.
- No new generic writer, plugin registry, public reconstruction wrapper, or
  arbitrary replay of processing recipes.

Exact reconstruction means the object represented by its successfully written
format, including explicitly requested encoding. It does not promise preservation
of arbitrary Python attrs, unsupported COG dimensions, arbitrary DataTree
hierarchies, or runtime cache/chunk choices. Unsupported inputs must fail before
the writer changes existing destinations. Existing native format limitations
remain explicit.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Native format readers/writers | Pixels, native raster metadata, resource lifetime, backend options |
| PySTAC and supported extensions | Item, Asset, temporal coverage, projection, bands, opening metadata |
| Selected STAC GeoParquet library | Item/table conversion, nested schemas, format metadata, serialization |
| GeoSave catalog module | Logical identity, asset composition, update validation, upsert, publication |
| GeoSave row accessor | Explicit result selection and application of a stored pixel window |
| Native GeoDataFrame | Querying, filtering, annotations, editing |

File metadata is authoritative. Catalog metadata describes it and supplies only
the structural information that the format cannot express. A native file must
not need a catalog to explain its ordinary grid, bands, or timestamps.

Do not infer object kind from `raster_metadata == {"image": ...}`, asset dictionary
order, parsed Item IDs, filenames, or directory scans. `to_raster` and `to_stack`
choose the result type explicitly. Group membership must be recorded explicitly
or provided by a native store hierarchy.

## Public reader contract

```python
def to_stack(
    self, *, layers: Collection[str] | str | None = None,
) -> xr.DataTree: ...

def to_raster(self, *, layer: str | None = None) -> xr.Dataset: ...
```

`to_stack` opens all logical layers when `layers=None`, or only the named layers.
A standalone Dataset record has one logical layer named `image`. Split COG band
assets belong to that logical layer; they do not become separate stack groups.

`to_raster(layer=...)` opens one logical layer, including its split bands and
recorded acquisitions. Omission works only when the row contains exactly one
logical layer. Missing layers raise `KeyError`; ambiguous selection raises
`ValueError` listing available layers. An empty selection raises clearly.

Rows remain selected through pandas. Keep table-level `raster_ids` and
`to_raster(id)` for logical Dataset acquisition catalogs. A multi-layer sample
is selected by its Item ID and opened as a stack; do not add a second stack query
language or table-level stack grouping in this change.

Both row readers apply a stored pixel window exactly once. They own all opened
resources, and close partially opened assets when reconstruction fails. Native
Dataset `to_array()` provides DataArray conversion; no row convenience wrapper
is needed.

## Records and assets

Retain the current Dataset acquisition model: one Item per COG acquisition;
one Item per complete Zarr/NetCDF raster. Preserve existing ID defaults and
filtered-acquisition selection. Filtering a complete-store Item selects that
store, not individual dates inside it.

A stack or training sample is one logical Item. Its layers may have independent
acquisition axes and a static label or DEM. Its temporal coverage describes the
dated layers; do not invent acquisition times for static layers. A completely
undated object can be saved without automatic STAC registration.

Each catalog asset must reference a physical file/store, with a correct media
type and any standard opening/group options needed. A directory of COGs must
not masquerade as one directly readable raster asset. A stack Item can contain
multiple physical COG assets for one logical layer. A grouped Zarr/NetCDF store
can expose group-selecting assets pointing to the same store.

Use standard STAC extension representations where they express the actual
information. The evidence checkpoint evaluates projection/band helpers,
Datacube dimensions/variables, and `xarray:open_kwargs` for backend/group options.
Do not add every extension automatically or mistake descriptive cube metadata
for a guaranteed lossless xarray codec.

Remove `geosave:format` if the tested media-type/opening contract replaces its
dispatch and format-conflict checks. Retain explicit logical grouping where
standard fields cannot express it without changing their meaning.

## Structural metadata decision

Before choosing new keys, produce a field ownership inventory covering current
`raster_metadata`, asset bands, `sources`, `geosave:raster`, `geosave:format`,
opening options, layer order, and temporal context.

For every value, record its consumer, native file representation, applicable
standard field, and whether a custom field is unavoidable. Check actual
round trips, including scalar time, a length-one time dimension, reordered dates,
mixed split dtypes, and static layers.

First try deleting fields and delegating their behavior to common conventions
or native storage. Only if an agreed invariant remains impossible, keep its
small structural contract with the writer and reader together. Do not scatter
nested dictionary construction across STAC, layout, reference, and training
modules. Do not replace the old dictionary with a larger typed schema, or
invent new field names before comparing the library-backed representations.

The replacement must keep metadata-only model context working:
`model.encoder.time.time_labels(row, raster=...)` currently consumes
`raster_metadata[raster]["times"]`. `transform.vector.from_layouts` copies those
values into tile references. Full frame order must remain available without
opening pixels, and cut-frame references must describe the cut frame rather
than the parent series. Snapshotting values into references may be legitimate;
it must reuse the same semantic extraction contract, not a second schema.

## Publication

All Dataset and Stack writers use one serialized publication policy:

1. Validate identity, schema, destination ownership, and format constraints.
2. Complete the native pixel write.
3. Reopen saved metadata and construct authoritative records.
4. Upsert by Item ID, retaining other Items and publish once.

`overwrite` controls pixel replacement. It must not mean replacing the entire
catalog. Failed or deferred pixel writes publish no new record. `compute=False`
must delay publication until the pixel task succeeds. Add matching `catalog`
and `id` options to Stack NetCDF writing if it joins the shared path.

Preserve relative href relocation, fsspec-backed catalogs, existing local atomic
catalog replacement, and existing non-atomic remote replacement behavior.
Concurrent catalog read/modify/write and transactions spanning raster files and
catalogs remain out of scope. Never delete unrelated files or imply a failed
publication rolled back already completed pixels.

## Library evidence checkpoint

Start with the installed library's native Item -> GeoDataFrame -> asset-opening
path and demonstrate it without any GeoSave fields. Use that as the reference
convention. Compare `stac-geoparquet` and `rustac` with identical real files and Items
only where observed behavior warrants an alternative.
Evaluate `rio-stac` for COG metadata and `xstac` for Dataset/datacube metadata
separately from serialization. Use disposable environments before changing
`pyproject.toml` or `uv.lock`.

Acceptance includes heterogeneous assets, later-appearing extension properties,
nulls, empty nested values, edited geometries/bboxes, reordered/subset frames,
microsecond and nanosecond timestamps, relative/remote hrefs, grouped stores,
static and independently dated layers, and resource/laziness behavior. Validate
Items against the declared STAC and extension schemas and test an independent
client opening direct COG assets. Record supported version floors.

Keep the existing library if it meets the contract with less GeoSave code.
Choose rustac only when the probes demonstrate a concrete improvement in the
required behavior or removal of bespoke conversion. API brevity alone is not a
reason to switch. If neither supports a requirement, record the narrow policy
and its test instead of silently dropping metadata or precision.

This checkpoint produces a representation decision and updates this spec before
schema-dependent implementation starts. Library selection and custom fields
are deliberately not settled by the agreed direction alone.

## Evidence already collected

On 2026-10-06, installed `stac-geoparquet 0.8.2`, PySTAC 1.14.3,
GeoPandas 1.1.3, and PyArrow 23.0.1 were inspected.

- Items -> Arrow -> Parquet -> Items preserved heterogeneous asset keys,
  nested properties, later-appearing properties, and nulls.
- Current GeoDataFrame writing preserved reordered rows and heterogeneous assets.
- Empty nested objects failed Parquet writing; nine-digit fractional STAC
  datetime strings failed parsing into `timestamp[us]`. These are observed
  library-path limitations, not proof that current native raster timestamps fail.
- Saving a second Stack COG wrote pixels before catalog `FileExistsError`;
  retrying with `overwrite=True` left only the second Item in the catalog.
- Existing focused tests passed: 46 passed, 1 deselected. The sandbox Zarr run
  stalled; the rerun with local runtime access completed in 5.97 seconds.

No product implementation or dependency changes were made during that review.
The model time consumer was discovered during planning, correcting the earlier
claim that stored time labels had no downstream consumer.

A subsequent smoke constructed a native PySTAC Item independently of GeoSave's
Item factory, with a real COG data asset and a thumbnail. Library Arrow
conversion, GeoVector concat/upsert, and row `to_raster()` opened it lazily with
the original grid, bands, pixels, and annotations. No GeoSave properties or
`raster_metadata` were supplied. The thumbnail was ignored. Table-level
`catalog.gs.to_raster()` still required `geosave:raster`, exposing the distinction
between ordinary Item opening and GeoSave's grouped acquisition reader.

## Sources

- [stac-geoparquet conversion and serialization](https://stac-utils.github.io/stac-geoparquet/latest/usage/)
- [rustac capabilities and Arrow interoperability](https://github.com/stac-utils/rustac-py)
- [rio-stac metadata generation](https://developmentseed.org/rio-stac/api/rio_stac/stac/)
- [xstac metadata generation](https://github.com/stac-utils/xstac)
- [Datacube extension](https://github.com/stac-extensions/datacube)
- [xarray-assets extension](https://github.com/stac-extensions/xarray-assets)

## Implemented representation decision

See [library evidence](../probes/2026-10-06-persistence-libraries.md) for the
selected field contract, dependency versions, discarded alternatives and known
limits. Keep PySTAC/stac-geoparquet; retain minimal temporal/order metadata and
explicit asset composition. Public readers return explicit native types.
