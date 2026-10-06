# Raster persistence indexed by a STAC GeoParquet catalog

Status: Dataset implementation complete and verified (1474 tests passed).
Stack grouping remains deferred.

## Caller API

```python
forest.gs.to_cog("samples", id="forest", catalog="samples/catalog.parquet")
water.gs.to_cog("samples", id="water", catalog="samples/catalog.parquet")

catalog = read_vector("samples/catalog.parquet")
catalog.gs.raster_ids                       # ["forest", "water"]
forest = catalog.gs.to_raster("forest")     # Dataset, all saved acquisitions

recent = catalog.loc[catalog["datetime"] >= "2025-06-01"]
recent.gs.raster_ids                        # Only IDs present in these rows
forest = recent.gs.to_raster("forest")      # Only selected acquisitions

forest_updated.gs.to_cog(
    "samples", id="forest", catalog="samples/catalog.parquet", overwrite=True,
)
```

The catalog is an ordinary GeoDataFrame. GeoSave adds discovery and opening
through its existing accessor, not another catalog wrapper.

```python
@property
def raster_ids(self) -> list[str]:
    """Return unique saved raster IDs in their first-appearance order."""

def to_raster(self, id: str | None = None) -> xr.Dataset:
    """Open one saved raster's selected acquisitions lazily."""
```

- `raster_ids` reads only the current table's `geosave:raster` column. It
  neither scans folders nor opens assets. An empty catalog returns `[]`.
- `to_raster(id)` selects that raster within the current table, merges bands
  within each acquisition, and concatenates acquisitions in chronological order.
- Omitting `id` works when exactly one raster remains. Zero or several raster
  IDs raise `ValueError`; an explicitly requested absent ID raises `KeyError`.
  Ambiguous-selection errors list the available IDs.
- A table lacking `geosave:raster` is not a grouped GeoSave raster catalog;
  these methods raise a clear error instead of guessing groups from Item IDs.
- Callers do not parse Item IDs or filenames. Normal pandas filtering remains
  the selection API; no second query language or special index is introduced.

## Saving and identity

`id: str | None = None` is optional on all three Dataset writers. When omitted,
use `raster.gs.anchor.stem` consistently for COG, NetCDF, and Zarr. An explicit
ID overrides that value. The destination filename does not determine identity.

```python
# Each example is an alternative format for the same logical raster.
forest.gs.to_cog("samples", catalog="samples/catalog.parquet")
forest.gs.to_zarr("samples/forest.zarr", catalog="samples/catalog.parquet")
forest.gs.to_netcdf("samples/forest.nc", catalog="samples/catalog.parquet")
# Default logical ID in each case: forest.gs.anchor.stem
```


```python
def to_cog(
    self,
    path: str | PathLike[str],
    *,
    id: str | None = None,
    split_bands: bool = False,
    map_scale: float | None = None,
    overwrite: bool = False,
    catalog: str | PathLike[str] | None = None,
    **options: Unpack[COGWriteOptions],
) -> Path:
    ...
```

For Dataset writes, `path` is the parent directory and `id` is the logical
raster ID. None uses `gs.anchor.stem`; callers updating a time series should
supply an explicit stable ID because its temporal extent can change. The ID
must be one safe path segment. The return value is the written directory or,
for an unsplit raster without a time axis, the single TIFF path.

One raster ID represents one grid and band schema. Appending incompatible
grids or band schemas under that ID raises; no implicit reprojection, dtype
conversion, alignment, or missing-band filling occurs.

For COG output, each acquisition is one STAC Item, hence one catalog row:

```text
id (unique Item identity)     geosave:raster    datetime
forest/20250601T103031        forest           2025-06-01T10:30:31Z
forest/20250611T103031        forest           2025-06-11T10:30:31Z
water/20250601T103031         water            2025-06-01T10:30:31Z
```

The writer generates Item IDs from the raster ID and the full normalized
acquisition timestamp. Timestamp formatting must retain sufficient precision
to avoid collisions; duplicate acquisition times are rejected before writing.
Item IDs do not depend on `split_bands`. STAC Item IDs are table keys, not
filesystem paths to be interpreted by readers.

`geosave:raster` is the single custom grouping property for Dataset catalogs.
It is stored in STAC `properties` and flattened by stac-geoparquet.

## Assets and file placement

For COG output, every data asset points directly to a COG, with its COG media type, data role,
and band metadata. No directory assets or custom `leaves` field are needed.

```python
# split_bands=True, one acquisition
assets = {
    "B04": {"href": "forest/20250601T103031/B04.tif", ...},
    "SCL": {"href": "forest/20250601T103031/SCL.tif", ...},
}

# split_bands=False, the same acquisition
assets = {"image": {"href": "forest/20250601T103031.tif", ...}}
```

The writer uses one arrangement; `layout=` is removed from Dataset COG saving.
Readers follow asset hrefs, so the arrangement is not a reconstruction rule.
Time-bearing Datasets use the paths above. Without a time axis, the paths are
`<id>.tif` or `<id>/<band>.tif`. Split bands retain their separate dtypes;
unsplit bands must share a dtype, as today.

Every file retains applicable CRS, transform, nodata, band names, attrs, and
precise time tags. Catalog metadata comes from the written storage snapshot.
Band order and whether the source had a scalar time or a time dimension must
survive through existing raster metadata; asset map order is not a band-order
contract. A missing time coordinate must not be invented on reopening.

STAC still requires a meaningful datetime or temporal interval. A raster with
neither cannot be cataloged automatically; saving files without `catalog=`
remains possible. No fake acquisition date is introduced.

Existing path rules remain: assets below the catalog directory use relative
hrefs; others use absolute hrefs. Reading resolves paths against the catalog
location. Moving a catalog together with its assets must keep it usable.

## Upsert and serialization

`catalog=` creates a table if absent and upserts incoming acquisition rows by
Item `id`, using the existing `gs.upsert(..., on="id")` behavior. Saving another
date adds a row; saving the same date replaces its row. Dates absent from a
later save remain. `overwrite` controls replacement of existing COG files;
this operation does not delete other dates or orphaned files.

Parallel workers collect Items and publish once. Concurrent read-modify-write
upserts to one Parquet file are not supported. Catalog publication happens
only after the new files and their metadata have been validated. Atomic local
catalog replacement remains; atomic replacement of a whole COG tree is deferred.

```python
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

# GeoSave provides domain values; PySTAC represents Items and assets.
items = [...]  # pystac.Item instances
stream = parse_stac_items_to_arrow(items, drop_invalid_properties=False)
to_parquet(stream, staging_path)
```

PySTAC and stac-geoparquet own STAC structure, Arrow conversion, and Parquet
format metadata. Remove manual `ITEM_COLUMNS`, `STAC_VERSION`, and the custom
STAC file-metadata stamping pass. GeoSave retains identity, domain metadata,
path resolution, upsert, and publication policy. Ordinary vector GeoParquet
writing continues through GeoPandas.

An edited GeoDataFrame can pass through Arrow to the library writer; the
library is not limited to Item inputs. GeoPandas can omit a covering `bbox`
column on read, so the write boundary must supply a correct STAC bbox for the
current rows. Never reuse an old positional bbox column after filtering,
reordering, upserting, or editing geometry. Round-trip tests verify nullable custom properties and heterogeneous assets
through the full write path.

## NetCDF and Zarr catalog indexing

```python
forest.gs.to_zarr(
    "samples/forest.zarr", id="forest", catalog="samples/catalog.parquet",
)
water.gs.to_netcdf(
    "samples/water.nc", id="water", catalog="samples/catalog.parquet",
)

catalog = read_vector("samples/catalog.parquet")
catalog.gs.raster_ids                         # ["forest", "water"]
forest = catalog.gs.to_raster("forest")       # Dataset from the Zarr store
water = catalog.gs.to_raster("water")         # Dataset from the NetCDF file
june = forest.sel(time=slice("2025-06-01", "2025-06-30"))
```

Both Dataset writers accept optional `catalog=` and `id=None`, with the same
`gs.anchor.stem` default, logical raster identity, and discovery API as COG. Zarr already exposes these
arguments; NetCDF gains them. Their destination remains the explicit store or
file path. Native format options, return paths, and `compute=False` remain.

Unlike separate COG acquisitions, a NetCDF file or Zarr store can contain the
whole time series. Index it as one Item with one direct asset and
`geosave:raster`. Its Item ID is the logical raster ID; no acquisition suffix
is needed. For a multi-date store, set `datetime=None` and the actual
`start_datetime`/`end_datetime`. Single-instant data can use `datetime`.
Time-free data still requires meaningful temporal metadata for STAC indexing.
Do not introduce per-date duplicate rows or custom time-slice asset selectors.

`to_raster(id)` dispatches through existing format readers and retains the
store's native time coordinates and lazy arrays. COG acquisition ordering
rules do not reorder a complete store's internal coordinates. Table filters
select entire indexed files/stores; they do not slice time inside an asset.
Use xarray `.sel(time=...)` for that. In mixed-format catalogs, interval rows
must be filtered by temporal overlap, not solely by the `datetime` column.

Saving a NetCDF/Zarr raster upserts its whole Item by stable ID and refreshes
its time range. `overwrite=True` replaces the destination, not individual
acquisitions inside it. This is not a store-append API. One catalog can contain
rasters stored in different formats, but one logical ID has one representation:
COG acquisition rows or one complete NetCDF/Zarr asset. Conflicting writes
raise before replacing assets; automatic format migration, multiple copies
under one ID, and multi-store shards are outside this slice.

With `compute=False`, catalog publication depends on successful completion of
the deferred pixel write; constructing the task must not publish a row. Reopen
the saved data with the necessary engine/group options to collect authoritative
metadata, then upsert and publish through the shared catalog writer. NetCDF
retains the chosen backend and group in the asset metadata for reopening. DataTree grouping remains the
separate stack decision below.

## Reading and resource ownership

The grouped reader opens only the selected data assets with explicit Dask
chunks, uses native raster readers to restore variable names, combines bands of an
acquisition, and concatenates times with exact spatial alignment. Metadata
merging uses existing attrs helpers. No pixel computation is needed to list
IDs or construct the returned Dataset; returned variables remain dask-backed.
The returned Dataset owns all opened resources and closes them together;
failures during opening close already-opened resources.

One selected acquisition from a time-dimensional raster keeps a length-one
time dimension. Grid, dtype, band, and time conflicts raise instead of being
silently coerced. Repeated Item rows, such as windowed sample references, are
not implicitly fused into one acquisition. Per-row window reading remains a
separate operation.

## Scope and remaining stack decision

This revision settles the Dataset catalog direction: list logical raster IDs,
select one, and open its Dataset from COG acquisition rows or a complete
NetCDF/Zarr asset.

The previous draft also proposed changing DataTree saving, renaming
`row.gs.to_xarray`, and removing directory readers across the package. Those
changes must not be carried forward mechanically: training sample rows can
currently represent several layers, and a stack can contain independently dated
layers and a static label or DEM.

Before extending this contract to DataTree, decide whether a stack is listed as
one saved ID with a layer selector or whether each layer is listed as a raster.
That choice also determines `to_stack` and the training catalog migration.
Keep these existing APIs unchanged in the Dataset implementation slice; this
is a scope boundary, not a compatibility alias. Directory-reader removal and
row-method renaming are deferred to that migration.

Zarr/NetCDF destination paths and DataArray's single-file COG API remain unchanged.
NetCDF/Zarr Dataset catalog indexing is in scope as described above.
Third-party STAC grouping, raster deletion, concurrent catalog mutation, and
whole-tree transactions are outside this slice.

## Verification

Required implementation tests mirror source modules:

- Identity: omitting `id` uses `gs.anchor.stem` for all three formats; an
  explicit ID wins, independent of destination filename.
- Discovery: unique IDs, first-appearance order, filtered and empty tables,
  missing grouping metadata, unknown and ambiguous selections.
- Real COG round trips: two rasters, multiple dates, split and unsplit bands,
  mixed split dtypes, band/time order, attrs, precise time, and scalar versus
  dimensional time. Assert no eager pixel reads while opening.
- Upsert: same acquisition replaces; new acquisition appends; other rasters
  and dates remain. Changing split mode preserves Item identity.
- Storage: relative hrefs survive moving the catalog with its assets; external
  assets remain absolute; no scan discovers obsolete unreferenced files.
- Serialization: valid PySTAC Items and STAC GeoParquet metadata; edited,
  filtered, reordered, and upserted frames preserve bbox, null properties,
  custom metadata, and differing asset keys through library round trips.
- NetCDF/Zarr: catalog discovery and lazy Dataset round trips, multi-date
  intervals, same-ID upserts, mixed-format catalogs with distinct raster IDs,
  unsupported same-ID representations, native group/backend options, deferred
  publication, and failed writes leaving the catalog unpublished.
- Failures: duplicate times, incompatible grids/bands/dtypes, missing files,
  and close behavior for partially opened acquisitions.

### Smoke evidence, 2026-10-06

Earlier in this planning session, stac-geoparquet 0.8.2 successfully wrote Items,
read them into GeoPandas, and wrote an edited frame through Arrow and the library
writer. Nullable properties and relative hrefs survived. Restoring the bbox
column was necessary; this corrects the old draft's claim that the library
writer only accepts Items and cannot support edited frames.

A separate throwaway probe passed with eight real COGs, two logical rasters,
two acquisitions each, and split uint16/uint8 bands. Four STAC rows yielded
`["forest", "water"]`; selecting forest reconstructed two dates with Dask-backed
variables, the original dtypes, and the expected pixels. Upserting one existing
Item retained four rows. This used the existing reader with explicit chunks
and band names from asset metadata, not the proposed accessor.

Initial probes exposed that the existing reader defaults to no Dask chunks and
did not expose the synthetic files under their band descriptions. The passing
probe supplied chunks and applied asset band names explicitly. The process
exited successfully but emitted `Error in sys.excepthook` messages during
shutdown; resource cleanup still needs dedicated implementation verification.
Neither probe establishes full metadata round-trip coverage.

Focused existing NetCDF/Zarr profile round-trip tests passed: 8 passed,
15 deselected. The sandbox run stalled at Zarr; the same selection passed
outside the sandbox. These verify existing persistence behavior, not the
proposed catalog integration, which still needs the tests listed above.

## References

- STAC Item specification: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md
- stac-geoparquet Arrow API: https://stac-utils.github.io/stac-geoparquet/latest/api/arrow/


## Raster reference cleanup (approved follow-up)

Persistent tiling and raster indexes are native GeoDataFrames. Specialized
constructors live at `geodata.transform.vector.from_layouts` and
`geodata.stac.item.from_assets` / `from_xarray`, not on GeoVector.
`assets` point at actual saved pixels; `source_assets` describes provenance.
A virtual tile carries the parent's assets and a pixel window. A materialized
tile points to its own saved pixels without that window. Prepared parents must
be saved before their virtual tiles can be reopened independently.

Training and validation partitions are separate Parquet files. No `split`
column is generated. Arbitrary annotations remain native dataframe columns.
Undated and pixel-only reference tables remain ordinary GeoParquet; asset href
relocation does not require STAC. STAC-shaped records use stac-geoparquet.

`row.gs.to_xarray()` opens stack layers or a logical acquisition's image layer;
`row.gs.to_raster()` merges an acquisition's bands. Both use the same native
asset opener and apply the row's optional window. `row.gs.crop(parent)` retains
the explicit prepared-parent path used during training. Catalog assembly uses
this row reader and joins acquisitions along time. It does not infer paths from
a directory convention or rename native raster bands. Raster metadata retains
band order, time-axis shape, and coordinate dtype.

COG leaf placement belongs to the layout writer. NetCDF and Zarr use their
native format writers. Catalog code validates updates, selects rasters and
publishes completed records; deferred writes publish after pixel completion.
