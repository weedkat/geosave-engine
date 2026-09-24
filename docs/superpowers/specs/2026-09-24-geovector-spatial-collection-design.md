# GeoVector spatial collection

## Intent

`GeoVector` becomes GeoSave's one in-memory representation for collections of
spatial records. A record is a geometry plus ordinary tabular properties. Its
geometry may be an irregular place, a label polygon, or the rectangular extent
of an xarray grid; those are values, not different row types.

The expanded type must support this flow without loading raster pixels into the
collection:

```python
bounds = prediction.gs.anchor.geobox.extent.to_crs("EPSG:4326").boundingbox
labels = read_vector("labels.parquet", bbox=tuple(bounds))
matches = labels.query(prediction, predicate="intersects")

record = GeoVector.from_xarray(
    prediction,
    path="rasters/prediction.zarr",
    fields=("time", "grid"),
)
catalog = GeoVector.concat([catalog, record])
catalog.to_geoparquet(
    "dataset/catalog.parquet",
    write_covering_bbox=True,
)
```

`Manifest` is removed. Its useful behavior becomes ordinary `GeoVector`
construction, keyed collection growth, GeoParquet persistence, and portable
asset references. The generated workspace's separate CSV/XLSX ingest ledger is
workflow state and is not part of this change.

## Design principles

- Geometry is the only required spatial value. A bbox and a polygon have the
  same status.
- A collection has one active CRS so GeoPandas can build and use one spatial
  index. Collection coordinates are never silently rewritten; relational
  operations may transform only a transient query or burn geometry.
- Pixel values stay in xarray or persisted raster assets. A vector record keeps
  only lightweight geometry, grid, time, and caller properties.
- Constructors return `GeoVector` instances. Collection growth composes those
  instances rather than introducing separate raster-record or manifest types.
- Native GeoPandas, pandas, Rasterio, and ODC operations do the spatial work.
  GeoSave owns conversion contracts and metadata preservation, not replacement
  geometry or table engines.
- Potentially eager raster/vector conversion is explicit.

## GeoVector state and invariants

`GeoVector` continues to wrap a `GeoDataFrame`. It may also remember the local
path it was read from solely to resolve relative asset references. That source
path is persistence context, not record data or collection identity.

The frame must have an active geometry column and CRS. Every present geometry
must be non-null, non-empty, and valid. An empty frame is permitted when its
geometry column and CRS are defined, enabling `GeoVector.empty(crs)` and empty
filtered reads. Operations that need a footprint raise a clear error for an
empty collection.

No identity is inferred, including for an anchor. The same geometry and time may
legitimately describe different assets, models, resolutions, or pixel phases.
Callers that need keyed replacement supply a meaningful property such as
`record_id` or `path`. Spatial and temporal selection are queries, not implicit
identity rules.

## Construction

All constructors produce normal, composable `GeoVector` values.

```python
GeoVector.empty(crs)

GeoVector.from_geometry(
    geometry,
    *,
    crs=None,
    **properties,
)

GeoVector.from_anchor(
    anchor,
    *,
    crs=None,
    fields=("time", "grid"),
    **properties,
)

GeoVector.from_xarray(
    data,
    *,
    geometry=None,
    crs=None,
    path=None,
    fields=("time", "grid"),
    **properties,
)
```

`from_geometry` creates one record. Bare Shapely, WKT, and GeoJSON geometry
keeps the existing WGS84 default; an ODC geometry retains its CRS. Properties
become columns.

`from_anchor` creates one record using the grid extent. `crs` optionally and
explicitly selects the collection plane; otherwise the extent stays in the grid
CRS. The result includes only the selected lightweight fields and caller
properties.

`from_xarray` accepts a geolocated `DataArray`, `Dataset`, or single-grid
`DataTree` and obtains its `GeoAnchor` through the existing accessor. Its default
geometry is the grid extent. Supplying `geometry` preserves a semantic polygon
instead, while grid columns still describe the actual xarray grid. A supplied
geometry follows `from_geometry`'s CRS rules before an explicit output `crs` is
applied.

`fields` is an explicitly typed collection of supported metadata groups.
`from_anchor` accepts `Literal["time", "grid"]`; `from_xarray` additionally
accepts `Literal["variables"]`:

| Field | Columns | Meaning |
| --- | --- | --- |
| `time` | `start_datetime`, `end_datetime` | Inclusive UTC coverage, null for timeless data |
| `grid` | `grid_crs`, `grid_transform`, `grid_height`, `grid_width` | Exact native pixel grid |
| `variables` | `variables` | Ordered xarray variable or band names |

Geometry and a supplied `path` are not field groups and are never silently
disabled. `fields=()` creates the smallest spatial record. Arbitrary xarray
attrs are not copied as a group: they may contain arrays, opaque objects, or
conflicting names. Callers select meaningful properties explicitly. A caller
property cannot replace a selected derived column.

Variable identity belongs to each native xarray accessor rather than a
`GeoVector` helper:

```python
array.gs.variables    # ("ndvi",) or () for an unnamed array
dataset.gs.variables  # ("red", "nir")
stack.gs.variables    # ("optical/red", "labels/class_id")
```

`from_xarray(..., fields=("variables",))` reads this common property. It does
not branch over the three xarray types itself.

## Collection growth

Literal concatenation and keyed replacement are separate operations:

```python
combined = GeoVector.concat([labels, anchors, raster_records])
updated = combined.upsert(new_records, on="path")
```

`concat` performs one native concatenation, unions property columns, fills
missing properties with nulls, preserves row order, resets the combined frame to
a `RangeIndex`, and retains duplicate rows. All non-empty inputs must have the
same CRS. A mismatch raises with instructions to call `to_crs`; no input is
silently transformed. Empty inputs participate when their CRS agrees.

`upsert` requires an explicit key column on both collections and non-null,
unique keys within the incoming collection. It replaces existing matching rows
and appends new keys, returning a new `GeoVector`. It does not treat geometry as
identity. The key is caller-owned; GeoVector never hashes geometry or anchor
metadata into one.

Both operations return new collections. Incremental callers should accumulate
small records and concatenate once rather than repeatedly copying a growing
frame. A mutable `add` API is intentionally omitted from the initial surface.

## Spatial selection

```python
matches = collection.query(target, predicate="intersects")
```

`target` may be a supported geometry, `GeoAnchor`, `GeoVector`, or geolocated
xarray object. The operation obtains its footprint, explicitly transforms that
footprint into the collection CRS as part of relating the two objects, uses the
GeoDataFrame spatial index for candidates, applies the requested exact GeoPandas
predicate, preserves source row order, and returns a `GeoVector` containing the
original geometries and properties.

The first implementation types `predicate` as a `Literal` containing
`intersects`, `within`, `contains`, `covers`, and `covered_by`. A runtime value
outside that set also fails before querying. An empty collection returns an
empty collection with the same schema and CRS.

For large persisted collections, filtering occurs in two stages:

1. `read_vector(path, bbox=..., columns=...)` asks GeoPandas/GeoParquet to read
   only candidate row groups and requested columns.
2. `query` applies the exact geometry predicate in memory.

GeoPandas can optionally write one bounding-box value per row using
`write_covering_bbox=True`. That extra GeoParquet 1.1 column enables the first
stage for WKB geometry, but it costs computation, increases the schema, and is
still marked experimental by GeoPandas. GeoVector does not enable it silently;
callers opt in when they intend to use filtered persisted reads. It is a file
query optimization, not a second boundary geometry or an in-memory R-tree. A
materialized `GeoVector` remains an in-memory GeoDataFrame; the design does not
claim out-of-core computation.

## Portable asset references

`path` is the canonical optional asset-reference column for the initial design.
It points to the raster or stack described by that row. It is not required for
plain labels or places.

When writing GeoParquet, absolute `Path` values are normalized relative to the
destination file's parent. Paths resolving outside that parent, including via
symlinks, are rejected before the destination is replaced. Relative path values
must also resolve within that parent. The stored value uses POSIX separators, so
moving the catalog and its asset directory together preserves references.

Reading a local vector file records its source path. `resolve_path(row)` accepts
one row mapping, resolves its relative `path` against the source file's parent,
and rejects escaped paths. It fails clearly when the vector has no source path
or the row has no asset. This contract is local-only; it contains no URI parsing
or remote asset branches. Multiple named asset columns remain out of scope until
a concrete caller defines them.

Catalog GeoParquet writes use a sibling staging file followed by atomic
replacement, preserving the behavior being absorbed from `Manifest`. Path
validation and serialization operate on a copy; a failed write does not mutate
the in-memory frame or replace the prior file.

## Flag raster conversion

Polygonization creates another ordinary collection:

```python
classes = GeoVector.vectorize(
    flags,
    value_name="class_id",
    mask=None,
    connectivity=4,
)
```

The input is one selected, two-dimensional, geolocated `DataArray`. Callers must
select time, band, or other extra dimensions first. Nodata is excluded by
default; an explicit mask may exclude additional pixels. Each contiguous region
becomes one row carrying its flag value. The output uses the raster CRS. Since
polygonization must inspect flag values, it computes a lazy input explicitly and
documents that eager boundary.

Rasterization uses an existing grid without inventing one:

```python
flags = classes.rasterize(
    like=reference,
    column="class_id",
    fill=0,
    dtype="uint8",
    all_touched=False,
)
```

`like` is a geolocated xarray object or `GeoAnchor`. Geometry is transformed to
the target grid CRS for the operation. `column=None` burns a boolean presence
mask; otherwise every value must be representable by the requested dtype. Later
rows win where geometries overlap, matching Rasterio draw order. The result is a
geolocated `DataArray` on the exact target grid. The initial implementation is
eager and allocates that grid; chunked rasterization is a later optimization,
not an alternate contract.

## Ownership and module changes

- `geodata/core/vector.py` owns the collection contract, constructors,
  composition, querying, asset resolution, and thin conversion methods.
- `geodata/core/array.py`, `raster.py`, and `stack.py` expose one shared
  `.gs.variables` contract appropriate to their native xarray object.
- `geodata/transform/vector.py` owns eager Rasterio polygonization and
  rasterization. GeoVector methods delegate to it so the useful fluent API does
  not make the collection class own conversion mechanics.
- Existing GeoParquet I/O continues to own format reading and writing. It gains
  source-path propagation, portable `path` normalization, staging, and atomic
  replacement. Covering bboxes remain an explicit native GeoPandas option.
- Existing xarray accessors continue to own anchor extraction. They do not store
  a `GeoVector` inside each raster.
- `geodata/pipeline/manifest.py`, its export, and its dedicated tests are removed.
  Manifest behavior is covered through `GeoVector` tests and public I/O.
- No new dependency, registry, schema language, raster wrapper, or compatibility
  alias is introduced.

## Code shape and typing

Public inputs use concrete types: Literal aliases for fields and predicates,
explicit unions for query targets, `DTypeLike` for raster output, and `object`
for caller-owned tabular properties. `Any` remains only at third-party
passthrough boundaries whose value vocabulary is not defined by GeoSave, such as
PyArrow keyword options.

A private helper is introduced only when more than one call site shares a real
domain operation. One-use normalization and dispatch stay inline, grouped into
short blocks with comments explaining the constraint. In particular, the
implementation has no `_anchor_id`, `_checked_fields`, `_variable_names`,
`_canonical_frame`, `_query_geometry`, `_polygonize_values`, `_target_geobox`,
or `_burn_values` methods. Existing nodata and dtype utilities are reused before
adding another helper.

## Errors and preservation

Operations fail without partial mutation for missing CRS, invalid geometry,
unsupported metadata fields, reserved-property collisions, incompatible
collection CRSs, missing or duplicate upsert keys, non-geolocated xarray input,
unsupported raster shape or dtype, and unsafe asset paths.

Querying and concatenation preserve original geometry and property values.
Registering xarray data reads coordinates and metadata only; it does not compute
pixel arrays. `vectorize` is the intentional exception because geometry depends
on pixel values. Rasterization does not alter the source collection.

## Acceptance examples and tests

Implementation starts with focused public smoke tests for these flows:

1. Build one-row vectors from geometry, an anchor, and xarray; concatenate them
   on one explicit CRS; verify property-column union and original geometry.
2. Query a large synthetic polygon collection with a prediction raster and
   return exactly the original intersecting rows in source order.
3. Register an xarray object using an irregular semantic polygon while retaining
   the rectangular native grid metadata.
4. Upsert a record by a caller-supplied key and prove its row is replaced rather
   than duplicated; concatenate the same records and prove duplicates are
   retained.
5. Save a catalog with an asset, reopen it after moving the directory, resolve
   the asset, and reject parent and symlink escapes without replacing prior data.
6. Opt into covering-bbox metadata, read with a bbox and selected columns, then
   apply an exact spatial predicate; verify the default does not add the column.
7. Vectorize categorical flags with nodata and disconnected regions, then
   rasterize the value column onto the original grid and compare valid pixels.
8. Verify registration of a Dask-backed raster does not compute pixels; verify
   vectorization does compute and documents that boundary.
9. Verify empty construction, empty filtered reads, CRS mismatch failures,
   invalid metadata groups, accessor variable names, and overlapping-
   rasterization row order.

Every touched source and test file must pass BasedPyright. Targeted mypy runs
ignore unavailable third-party stubs but must report no code errors. Repository-
wide type-check failures outside this change are recorded separately rather than
hidden or expanded into this feature.

After focused tests pass, run the geodata suite and the full project suite. The
existing working tree contains unrelated changes; implementation and commits
must include only files belonging to this design.

## Breaking changes and limits

`geosave_engine.geodata.pipeline.Manifest` is removed without a compatibility
alias. Callers construct or read `GeoVector` collections and use `concat` or
`upsert` instead. No automatic `anchor_id` is produced. `GeoVector` begins
accepting a correctly typed empty frame, and GeoParquet catalog writes gain
portable-path validation; covering bboxes are opt-in through GeoPandas' native
option.

The first implementation is single-process and local-path oriented. A loaded
collection is memory-resident, atomic replacement assumes one writer, and
vectorize/rasterize are eager. Partitioned catalogs, concurrent writers, URI
asset resolution, chunked conversion, and multiple named assets require future
use cases and are not implied by this interface.
