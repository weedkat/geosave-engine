# GeoVector foundation: plain vectors, Items from rasters, a Parquet item table

Supersedes `2026-10-07-geovector-conventions.md`,
`2026-10-07-catalog-group-naming-design.md` and the plans
`2026-10-07-geovector-conventions.md` and `2026-10-07-catalog-raster-indexing.md`.
Tiling and the chip index are out of scope and get their own spec.

## Problem

The `gs` GeoDataFrame accessor does three unrelated jobs: feature operations,
a STAC item table, and raster loading. The symptoms:

- `GeoVector.normalize` adds a `datetime` column to every frame, including a
  plain plots file.
- GeoParquet and GeoJSON I/O pick their format by sniffing for a
  `stac_version` column; `read_vector` sniffs for `assets`.
- Items are built by writing a raster, reopening every written file, and
  regrouping the files into scenes by parsing timestamps back out of them.
- Four `geosave:*` fields exist only so a table row can be turned back into a
  Dataset or stack.

## Definitions

| Thing | Definition | Owner |
| --- | --- | --- |
| GeoVector | A GeoDataFrame with an active geometry column and a CRS. No required columns. | `geodata/core/vector.py` |
| Item | A `pystac.Item` describing one saved raster. It exists only once the raster has a path. | `geodata/stac/item.py` |
| Item table | Items stored as stac-geoparquet, read back as a GeoVector that happens to have `id`, `datetime` and `assets` columns. | `geodata/stac/table.py` |
| GeoAnchor | Where, when and on which grid. The only object passed between vector and raster. | `geodata/core/anchor.py` |

Rules that follow:

1. A `gs` method on a GeoDataFrame works on every GeoDataFrame.
2. Anything that needs STAC columns is a function in `geodata/stac` that takes
   the table.
3. `geodata/io` knows file formats and nothing about STAC.
4. Writers return paths. Items are built from a raster and those paths.
5. One Item is one raster: one group of a stack, and for COGs one scene.
6. A stack is composed in memory with `stack(...)`. It is never one row.
   Rows saved from one stack share the `geosave:stack` property, and a
   group's name is its Items' `collection`.

## Flow

```python
# In memory -> saved -> Items
paths = ds.gs.to_cog("data/forest")
items = ds.gs.to_items(paths)

# Saved earlier -> Items
items = read_raster("data/forest.zarr").gs.to_items()

# Stack: any stack writer, one Item per group (per scene for COGs)
items = sample.gs.to_items(sample.gs.to_zarr("data/s0.zarr"), name="s0")
items = sample.gs.to_items(sample.gs.to_cog("data/s0"), name="s0")

# Items -> table on disk -> rows -> raster
stac.table.write(items, "data/catalog/part-0.parquet")
rows = stac.table.read("data/catalog", bbox=bounds)
rows = rows.gs.query(anchor)
ds   = stac.table.load(rows)

# Plain vectors never touch any of the above
plots  = read_vector("plots.geojson")
anchor = GeoAnchor.from_geometry(plots.gs.footprint, resolution=10)
mask   = plots.gs.rasterize(ds, column="class")
polys  = prediction.gs.vectorize()
```

## Module responsibilities

### `geodata/core/vector.py` (GeoVector)

Keeps: `crs`, `footprint`, `query`, `rasterize`, `to_geojson`,
`to_geopackage`, `to_geoparquet`, `concat`, `from_geometry`.

Removes: `normalize`, `timespan`, `to_anchor`, `to_items`, `from_items`,
`upsert`, `to_raster`, `to_stack`, `raster_ids`, `stack_ids`, `empty`,
`vectorize`.

`query(target, predicate=)` filters by space, and also by time only when the
target has a timespan and the frame has a `datetime`, `start_datetime` or
`end_datetime` column. Undated rows always pass the time filter. Missing
columns are never created.

### `geodata/core/array.py`, `raster.py`, `stack.py`

- `GeoArray.vectorize(value_name=, mask=, connectivity=)` replaces
  `GeoVector.vectorize`. It delegates to `transform.vector.vectorize`.
- `GeoRaster.to_items(paths=None, *, collection=None, datetime=None)` never
  writes and never opens a file. `paths` is what `to_cog`, `to_zarr` or
  `to_netcdf` returned. `None` uses the single file or store the raster was
  read from, and fails for a raster that was not read from one.
- `GeoStack.to_items(paths, *, name, datetime=None)` takes what a stack writer
  returned: a mapping of group to files, or one store path. `name` is the
  stack's identity and is required. There is no `collection` argument: each
  group's name is its collection.
- `GeoArray.to_items(paths, *, collection=None, datetime=None)` delegates to
  its one-variable raster.
- An Item's id is its identity in the table; the file name is a location kept
  in the asset href. `to_items(..., id="{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}")`
  fills a `GeoAnchor.format` template from each Item's own raster. Without
  `id`, the id defaults to the saved file's name. Two Items of one call
  getting the same id is an error.
- The five `to_items` overloads and their writer options go away.

### `geodata/stac/asset.py`

- `from_raster(raster, href)` builds one Asset from a raster's grid, bands and
  time. The media type comes from the href suffix.
- `default_key(raster)` stays.
- `from_path` and the `driver` parameter are deleted.

### `geodata/stac/item.py`

- `from_raster(raster, hrefs, *, id, collection=None, datetime=None)` builds
  one Item. Geometry and bbox come from the raster's geobox, time from its
  assets, assets from `hrefs` (asset key to href). `datetime` is one instant
  or a `(start, end)` pair.
- `from_files(raster, paths, *, collection=None, datetime=None)` pairs a
  writer's returned paths with the raster and returns a tuple of Items.
- `from_stack(rasters, paths, *, name, datetime=None)` builds every group's
  Items and stamps `geosave:stack`.
- `from_assets` is deleted.

`collection` is never invented. Without `collection=`, a raster's Items carry
none, and a table built only from such Items has no `collection` column.

Pairing rule in `from_files`:

| Paths | Meaning | Items |
| --- | --- | --- |
| one `.zarr` / `.nc` | one store | one Item, asset key `default_key(raster)` |
| as many `.tif` as instants | one file per scene | one Item per instant |
| instants × variables `.tif` | split bands, time then variable order | one Item per instant, one asset per variable |
| any other count | error | none |

A raster without a time dimension counts as one instant.

Defaults:

| Source | `id` | `collection` | `geosave:stack` |
| --- | --- | --- | --- |
| DataArray or Dataset, store | store stem | `collection=`, else absent | absent |
| DataArray or Dataset, COG scene | scene name the writer used, for example `forest_20250601T103031` | `collection=`, else absent | absent |
| DataTree group in a store | `<name>/<group>` | group name | `name` |
| DataTree group as COGs | `<name>/<scene name>` | group name | `name` |

A timeless raster has no time for its Item, so `from_raster` raises unless
`datetime=` is given. Inside a stack, a timeless group takes the span of the
stack's dated groups; a stack with no dated group needs `datetime=`.

A group inside a grouped store is recorded on its asset as
`xarray:open_kwargs = {"group": "<name>"}`, the published `xarray-assets` STAC
extension. This replaces `geosave:group`.

### `geodata/stac/table.py` (new)

- `from_items(items) -> GeoDataFrame` converts in memory through
  stac-geoparquet.
- `write(items, path, *, overwrite=False, storage_options=None)` stores hrefs
  relative to the table, then writes with the plain `io.geoparquet.write` and a
  covering bbox. There is no second Parquet writer.
- `read(path, **options) -> GeoDataFrame` reads with the plain
  `io.geoparquet.read` (one file or a directory of part files), makes hrefs
  absolute and drops the null asset entries Parquet adds. `bbox`, `columns` and
  `filters` are pushed down by GeoPandas.
- `load(rows, *, assets=None, **options) -> Dataset` passes the selected
  assets' hrefs to `read_raster`, with `chunks={}` unless overridden and any
  `xarray:open_kwargs` forwarded. It raises `KeyError` for a frame without
  `assets` or a named asset that no row has.

Rows to Items is not provided: nothing needs it once STAC JSON writing is gone.

Adding to a catalog is writing another part file. There is no upsert.

### `geodata/io`

- `catalog.py` is deleted.
- `geoparquet.py` reads and writes plain GeoParquet only, losing its
  stac-geoparquet branch. `relative_hrefs` and `absolute_hrefs` move to
  `stac/table.py` as private helpers.
- `geojson.py` reads and writes plain GeoJSON only. STAC ItemCollection JSON is
  read with `pystac.ItemCollection.from_file` and `stac.table.from_items`.
- `geopackage.py`, `geoparquet.py`, `geojson.py` lose `datetime_column`.
- `readers.read_vector` no longer looks for `assets`.

### `geodata/conventions.py`

`RASTER_ID`, `GROUP` and `GROUPS` are deleted. `STACK_ID`
(`geosave:stack`) moves to `geodata/stac/item.py`, the only module that
writes it.

### Callers outside `geodata`

- `workflow/tasks/labels.py`: the directory branch builds each Item with
  `stac.item.from_raster(read_raster(path), {"label": path}, id=sample_id)`;
  the table branch reads with `stac.table.read`.
- `ml/segmentation/supervised/data.py`: the manifest is an item table with one
  row per group, read with `stac.table.read`. Each sample is the rows sharing
  one `geosave:stack` value, composed with `stack({group:
  stac.table.load(rows of that collection)})`. Group names must cover the model
  spec's rasters and the target. The chip reference keeps `source_id` (the
  stack name) and copies annotation columns from the sample's target row;
  `source_assets` is dropped, since a join on `geosave:stack` recovers it.
- `core/anchor.py` keeps using `GeoVector.from_geometry`.
- `docs/guides/architecture.md`, `model/README.md`, the `io` package docstring
  and any workspace template using the removed calls are updated.

## Evidence from smoke tests

| Claim | Result |
| --- | --- |
| Items built from the in-memory raster match today's | Zarr identical; COG identical except `eo:bands[].description` |
| Building Items opens no file | 0 files reopened |
| A reopened raster gives the same Item without loading pixels | confirmed for Zarr and COG |
| GeoPandas reads a stac-geoparquet file with bbox and datetime pushdown | 50,000 items, 0.04 to 0.12 s, CRS WGS84 |
| A directory of part files reads as one table | confirmed with `gpd.read_parquet` |
| One group of a grouped Zarr or NetCDF store reads as a raster | confirmed with `read_raster(path, group=...)` |
| An item table written by the plain GeoPandas writer is valid stac-geoparquet | stac-geoparquet reads it back to identical Items; bbox pushdown works |

Libraries rejected: `xstac` (no release since 2023), `rio-stac` (weaker than
`stac/asset.py`, maintenance only), `xpystac` (thin, failed on a COG
ItemCollection), `rustac` (works, but GeoPandas already covers the query).

## Decisions taken by default

Each is a choice made for you; say so if you want the other option.

1. **Timestamp precision.** `from_items` drops today's
   nanosecond-restoring workarounds and keep stac-geoparquet's microseconds.
2. **Media type of `.tif`.** Recorded as GeoTIFF, not "cloud optimized",
   because telling them apart requires opening the file.
3. **`GeoVector.empty` and `GeoVector.timespan`** are removed as unused
   outside the removed code.

## Known gaps, deferred

- **Chip windows.** Reading pixel windows from table rows (`row_off`,
  `padding`, ...) disappears with `catalog.py`. `chip_windows` and
  `crop_record` stay and keep serving the Dataset directly.
- **Remote catalog writes** go through the existing `io.geoparquet.write` and
  were not smoke-tested here.

## Follow-up, separate plan

Remove `StacMetadata`: the xarray object keeps only what is needed to
interpret its pixels, and source history belongs in the manifest. This
touches `attrs/models/stac.py`, `attrs/headers/stac.py` and the
`item_properties` and `asset_fields` options of `StacSourceConfig`.

## Tests

Tests mirror the source tree.

| File | Covers |
| --- | --- |
| `tests/geodata/stac/test_item.py` | `from_raster`, `from_files` pairing table, defaults, timeless error |
| `tests/geodata/stac/test_asset.py` | fields from grid, bands, time; media type by suffix |
| `tests/geodata/stac/test_table.py` (new) | Items to Parquet to rows round trip, with the file readable by stac-geoparquet; relative hrefs; part directory; bbox pushdown; `load` is lazy and matches the written pixels; grouped store group |
| `tests/geodata/core/test_vector.py` | plain frames are not modified by reading; `query` with and without time; removed methods are absent |
| `tests/geodata/core/test_raster.py`, `test_stack.py` | `to_items(paths)` computes no pixels; `vectorize` on the array accessor |
| `tests/geodata/io/test_geojson.py`, `test_geoparquet.py`, `test_readers.py` | plain round trips; no `datetime` column added |
| `tests/workflow/tasks/test_labels.py`, `tests/ml/segmentation/supervised/test_data.py` | updated callers |

`tests/geodata/io/test_catalog.py`, `test_catalog_records.py` and
`tests/geodata/core/test_catalog.py` are deleted; their round-trip cases move
to `test_table.py`.

## Breaking changes

Alpha: no compatibility aliases. Removed public names: the GeoVector methods
listed above, `io.catalog`, `stac.asset.from_path`, `stac.item.from_assets`,
the `driver` and writer options of `to_items`, `datetime_column` on vector
readers, and the `geosave:raster`, `geosave:group` and `geosave:groups` fields.
The training manifest changes from one row per sample to one row per group. Existing catalogs carrying those fields
still read; the fields become ordinary columns.
