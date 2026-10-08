# STAC Through PySTAC Classes Design

**Status:** Implemented 2026-10-08; plan in
`plans/2026-10-08-stac-pystac-classes.md`. Supersedes
`2026-10-08-stac-extension-modules-design.md` and the plans
`2026-10-08-stac-extension-modules.md` and `2026-10-08-stac-table-client.md`.

**Goal:** `xarray -> pystac.Item -> STAC GeoParquet` built the way each
library intends: PySTAC extension classes instead of key insertion,
stac-geoparquet for Item <-> table conversion and writing, `rustac` for
searching, and GeoSave's own `gs` properties instead of re-derived facts.

## Rules

1. **No key insertion.** Nothing outside an extension class writes
   `extra_fields[...]` or `properties[...]`.
2. **PySTAC has the class: use it.** `X.ext(obj, add_if_missing=True).apply(...)`
   writes the fields and maintains `stac_extensions`.
3. **PySTAC has no class and we need the extension: write ours on PySTAC's
   base** (`PropertiesExtension` + `ExtensionManagementMixin`), one file per
   schema in `stac/extensions/`. Callers use it exactly like a built-in.
4. **Nothing reads it: do not publish it.** CF and Datacube go.
5. **Items and item tables convert in two functions.** `table.from_items`
   (stac-geoparquet's `parse_stac_items_to_arrow`) and `table.to_items`
   (`rustac.from_arrow`) are the one way between `pystac.Item` and a
   GeoDataFrame. No other code reads or writes a table cell. `GeoVector`
   stays a general vector accessor and knows nothing of STAC.
6. **Facts come from `gs`.** Footprint from `gs.geobox`, time from
   `gs.times` and `gs.timespan`, ids from `gs.anchor.format`, attrs from `gs.attrs`.

## Caller view

```python
paths = tree.gs.to_cog("samples/s0")                     # {group: files}, unchanged
items = stac.create_stack_items(paths, name="s0")        # group name = collection id
stac.table.write(items, "catalog/items.parquet", collections=[optical, label])

rows = stac.table.read("catalog/items.parquet")          # GeoDataFrame, hrefs openable
picked = rows[rows["collection"] == "optical"].gs.query(anchor)
ds = stac.table.load(stac.table.to_items(picked))        # lazy Dataset

# a new batch: concat like any vector, write again
more = stac.table.from_items(new_items)
stac.table.write(
    GeoVector.concat([rows, more]),
    "catalog/items.parquet",
    collections=stac.table.read_collections("catalog/items.parquet").values(),
    overwrite=True,
)
```

The manifest is the table: a raster is written, its paths become Items, and
the Items become rows. A table is one file.

A DataTree is saved as one collection per group; one file holds every group of
every sample. A sample's rows are found again through `geosave:stack`.

## Layout

```text
geodata/
├── attrs/
│   ├── headers/stac.py     create_header(...)   Items -> AttrsHeader, for catalog loads
│   ├── extensions/         DELETED
│   └── models/             unchanged
└── stac/
    ├── extensions/         one module per schema, both directions through its PySTAC class
    │   ├── projection.py       write(asset, raster)
    │   ├── raster.py           write(asset, raster)   read(band) -> Packing, CFVariable(units)
    │   ├── eo.py               write(asset, raster)   read(band) -> Spectral, CFVariable(long_name)
    │   ├── classification.py   write(asset, raster)   read(band) -> Legend
    │   ├── zarr.py             ZarrExtension + write(asset, raster)
    │   └── geosave.py          GeosaveExtension   geosave:stack (tile fields join it later)
    ├── item.py             create_collection, create_item, create_items, create_stack_items
    ├── table.py            from_items, to_items, write, read, read_collections, load
    └── client.py           StacTableClient over rustac's DuckdbClient, for StacSource
```

Deleted: `attrs/extensions/`, `stac/asset.py`,
`stac/extensions/{types,datacube,cf}.py`, the `BAND` tuple, both Protocols,
`schemas()`, `band_fields`, `_upgrade`, `_describe`, `_join`, `absolute_href`,
`_relative_hrefs`, `_absolute_hrefs`, `_data_assets`, `DataAsset`, `_Saved`,
`_timespan`.

## Extensions used

| Fact | Class | Version | Level |
| --- | --- | --- | --- |
| grid | `pystac.extensions.projection.ProjectionExtension` | v2.0.0 | asset |
| stored dtype, nodata, packing, unit | `raster.RasterExtension`, `RasterBand` | v1.1.0 | asset `raster:bands` |
| spectral facts, band names | `eo.EOExtension`, `Band` | v1.1.0 | asset `eo:bands` |
| legend | `classification.ClassificationExtension`, `Classification` | v2.0.0 | on the `RasterBand` |
| Zarr store metadata | ours, `ZarrExtension` | v1.1.0 | asset |
| stack identity | ours, `GeosaveExtension` | v0.1.0 | item properties |

PySTAC 1.14.3 has no class for the STAC 1.1 `bands` list, so GeoSave publishes
the v1.1.0 listings its classes model. odc-stac 0.5.2 reads dtype, nodata and
unit from `raster:bands` (smoke-tested), which it does not from `bands`.

## Writing: one module per extension

```python
# stac/extensions/projection.py
def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write a raster's grid as Projection fields."""
    geobox = raster.gs.geobox
    ProjectionExtension.ext(asset, add_if_missing=True).apply(
        code=..., shape=list(geobox.shape), transform=list(geobox.transform)[:6]
    )

# stac/extensions/__init__.py
# Classes sit inside the bands the Raster extension stores, so it comes first.
ASSET = (projection, raster, eo, classification, zarr)

# stac/item.py
item.add_asset(key, pystac.Asset(href, media_type=..., roles=["data"]))
for extension in extensions.ASSET:
    extension.write(item.assets[key], raster)
```

A module reads the raster through `gs` (`gs.geobox`, `gs.attrs`,
`gs.variables`) and writes through its schema's PySTAC class. A module whose
facts the raster lacks writes nothing and declares nothing.

- **raster:** one band per variable, in file order. `data_type` is
  `DataType(dtype.name)`, or `DataType.OTHER` for a dtype the enum lacks
  (`bool`). A non-finite fill is a `NoDataStrings` member.
- **eo:** one band per variable, named after it, only when a variable carries
  `Spectral`. A label or a DEM declares no EO; its names stay in the file.
- **classification:** each `Legend` becomes classes on its variable's Raster
  band, read back from `RasterExtension.ext(asset).bands`, so the Raster
  module must have run (tested). A legend with `flag_masks` writes nothing. A
  class name outside `[0-9A-Za-z-_]+` raises `ValueError` naming it.
- **zarr:** written where the raster was opened from a Zarr store.

Adding an extension is one module and one entry in `ASSET`.

## Reading: the same modules, composed by one header factory

Reading STAC into attrs is needed in one place only: `StacSource.load`, which
loads pixels from a catalog through odc-stac. A catalog's files often do not
carry their packing, units or spectral facts, and odc-stac passes on dtype,
nodata and units but not scale and offset, so the Item is the only source for
them. GeoSave's own tables do not need it: `table.load` opens the files, and
the files carry their attrs.

Each module that has something to say on read owns it beside its `write`:

```python
raster.read(band: RasterBand) -> list[AttrsModel]          # Packing, CFVariable(units)
eo.read(band: Band) -> list[AttrsModel]                    # Spectral, CFVariable(long_name)
classification.read(band: RasterBand) -> list[AttrsModel]  # Legend
```

`attrs/headers/stac.py` holds `create_header(items, collection, loaded, ...)`,
the header factory `StacSource` rebases from, beside `headers/gdal.py` and
`headers/geobox.py`. It imports the extension modules inside `band_attrs`,
not at module level, because the STAC package imports this module while it
loads; importing `geodata.attrs` alone still loads no STAC code (tested). `read_bands(asset)` returns one (`RasterBand`, EO
`Band`) pair per band and `band_attrs` merges what the three modules read from
it. Foreign Items often carry `raster:bands` without declaring the schema, so
`read_bands` uses PySTAC's concrete `AssetRasterExtension(asset)` and
`AssetEOExtension(asset)`, which do not require the declaration and do not
change the Item. A catalog publishing STAC 1.1 `bands` (PySTAC has no class
for it) is read by splitting each band into those same two classes; that
split is the one keyed read in the package. A `description` on a `raster:bands`
entry is no longer read; the Raster extension defines none, the EO band does.

`read_asset_fields` returns the asset's fields as published, without the band
listings, overlaid with the selected `RasterBand` and EO `Band` `to_dict()`.
`StacSourceConfig.asset_fields` therefore names v1.1.0 keys (`scale`,
`center_wavelength`).

## Our extension classes

```python
# stac/extensions/zarr.py
class ZarrExtension(PropertiesExtension, ExtensionManagementMixin[pystac.Item]):
    """Zarr store metadata on one asset."""

    name: Literal["zarr"] = "zarr"

    def apply(self, *, zarr_format: int, node_type: str, consolidated: bool) -> None: ...

    @property
    def zarr_format(self) -> int | None: ...          # zarr:zarr_format

    @classmethod
    def get_schema_uri(cls) -> str: ...

    @classmethod
    def ext(cls, asset: pystac.Asset, add_if_missing: bool = False) -> ZarrExtension: ...
```

`GeosaveExtension` has the same shape over `item.properties` with one property,
`stack`. Its schema is committed at the repository root,
`schemas/geosave/v0.1.0/schema.json`, and its URI is that file's raw GitHub
URL, so it resolves once the file is on `main` with no site to deploy. A
version's folder is never edited after release; the later tile extension adds
fields as a new version folder.

Smoke-tested: an Item carrying `ZarrExtension` fields validates against the
published Zarr schema and reads back through the class after `rustac.search`.

## Building Assets, Items and Collections

PySTAC builds the objects in its documented order; GeoSave adds no builder of
its own beside it.

```python
item = pystac.Item(id, geometry, bbox, datetime, {}, collection=collection_id)
item.add_asset(key, pystac.Asset(href, media_type=..., roles=["data"]))
for extension in extensions.ASSET:
    extension.write(item.assets[key], raster)
```

The asset is added before the modules write because an extension declares its
schema on the asset's owner; `.ext(asset, add_if_missing=True)` raises on an
asset with no Item (tested).

A Collection is `pystac.Collection(...)` with `pystac.Extent.from_items(items)`
and `item.set_collection(collection)`. PySTAC's catalog route
(`collection.add_items`, `normalize_hrefs`, `save`) builds a tree of JSON
files linked by parent and root links, which a table does not have, so it is
not used.

`item.py` keeps its four public functions and their signatures. Its internals
are rewritten on `gs`:

- **Footprint:** union of each raster's `gs.geobox.geographic_extent`.
- **Time:** rasters holding one label between them set `datetime` to that
  label, read from `gs.times` (fixed to read a scalar `time` coordinate). Rasters holding several set `start_datetime`
  and `end_datetime` from `gs.timespan`. `datetime=` from the caller covers
  timeless rasters. No reconciliation between the two.
- **Scenes:** files are grouped by their `gs.timespan`, in the order given.
  Files sharing a span are one Item; a store has its own span and is its own
  Item. No `("store", index)` sentinel.
- **Ids:** `gs.anchor.format(id)` when a template is given, else the file stem
  or the folder several files share.
- **Collection:** passed straight to `pystac.Item(collection=...)`.
- **Stack:** `GeosaveExtension.ext(item, add_if_missing=True).stack = name`.

Opened rasters are closed through one `ExitStack`; no pixel is read.

## Table

```python
# stac/table.py
def from_items(items) -> GeoDataFrame
def to_items(rows) -> list[pystac.Item]
def write(items, path, *, collections=None, overwrite=False, storage_options=None) -> Path | str
def read(path, **options) -> GeoDataFrame
def read_collections(path, *, storage_options=None) -> dict[str, pystac.Collection]
def load(items, *, assets=None, **options) -> Dataset
```

**from_items / to_items.** `stac_geoparquet.arrow.parse_stac_items_to_arrow`
into `GeoDataFrame.from_arrow`, and `rustac.from_arrow` over the rows' Arrow
table into `pystac.Item.from_dict`. Each direction uses the library that
handled it in testing: stac-geoparquet's table states WGS84 and typed
datetimes, where rustac's stated no CRS; rustac converts rows filtered down to
one collection, where stac-geoparquet raised on the other collection's
all-null asset column. These two functions are the only code that knows a
row's shape, and they own its one wart: geopandas drops the `bbox` covering
column when it reads a file, so `from_items` drops it too and `to_items`
rebuilds it from the geometry's bounds, which is what a STAC bbox is.

`from_items` resolves relative asset hrefs for an Item that carries a self
href, since a row keeps none to resolve against later.

**write.** Takes Items or rows; rows become Items through `to_items`. Items
are cloned, given the table as self href, and `make_asset_hrefs_relative()`,
so a table and its assets move together. With `collections`, each is cloned,
given `Extent.from_items` of its rows and the table as self href, and set on
its Items with `set_collection`, which adds the collection link the Item
schema requires. `stac_geoparquet.arrow.to_parquet(table, target,
collections=...)` writes the file through `storage.write`, which keeps the
suffix, overwrite and remote checks; the Collections sit in the file's
`stac-geoparquet` metadata. A row naming a collection absent from
`collections` raises `ValueError`. Without `collections` none is stored and
Items carry the id only.

**read.** `read_vector`; a file with no `stac_version` column is a caller's
own table, such as a hand-made label table, and is returned as read.
Otherwise rows go through `to_items`, each Item's hrefs made
absolute against the table, and back through `from_items`, so a row's href
opens directly.

**read_collections.** Reads the file's `stac-geoparquet` metadata key and
builds each `pystac.Collection.from_dict`. Empty where the file stores none.
Neither library reads this key back, so this small reader is ours.

**load.** Takes Items, keeps assets whose roles are absent or include `data`,
and hands `asset.get_absolute_href()` to `read_raster`.

**Adding Items.** A Parquet file is not extended in place. A new batch is
concatenated onto the rows with `GeoVector.concat` and written again, as any
vector is. Collections live in the file being replaced, so the rewrite passes
them again.

```python
# stac/client.py
class StacTableClient:
    def __init__(self, path, *, duckdb=None) -> None
    def search(self, query) -> list[pystac.Item]
    def collections(self) -> set[str]
    def collection(self, collection) -> pystac.Collection
    def source(self, collection) -> StacSource
```

`StacTableClient` exists so a `StacSource` can load from a table as it does
from an API. `search` calls `DuckdbClient.search` on the file, builds each
`pystac.Item.from_dict`, sets the table as its self href and calls
`make_asset_hrefs_absolute()`. `duckdb` stays: a bucket behind a custom
endpoint is only reachable through a session the caller configured with
`CREATE SECRET`. `collection(id)` returns the Collection the file stores, read
through the same session, else the one rustac derives from the rows.

Smoke-tested 2026-10-08, locally:

- Items -> Arrow -> GeoDataFrame -> `GeoVector.concat` with a table from
  `read_vector` -> write -> `read_vector` -> filter -> Arrow -> Items. Optical
  and label rows with different asset keys concatenated; the typed classes
  read scale and classes back.
- `to_parquet(collections=...)` stores Collections synchronously; rustac
  searches and filters that file.
- Rows read by `read_vector` have no `bbox`, and both stac-geoparquet and
  rustac refuse to convert them until it is rebuilt.
- Relative asset hrefs resolve through PySTAC's own href methods.
- An Item whose collection link points at the table passes `Item.validate()`;
  with a collection id and no link it fails.
- `rustac.GeoparquetWriter.add_collection` is async only and fails inside a
  running event loop, and `DuckdbClient.get_collections` ignores stored
  Collections.

Smoke-tested 2026-10-08, against an S3 bucket behind a custom endpoint:

- `storage.write` with `to_parquet` wrote the table; a second write without
  `overwrite` raised `FileExistsError`.
- The stored Collection read back from the remote file's metadata.
- PySTAC made asset hrefs relative to an `s3://` table.
- `rustac.search` with credentials as keyword arguments or environment
  variables ignored the endpoint and failed; `DuckdbClient` with
  `CREATE SECRET` searched and filtered the table.

## Consumers

- `ml/segmentation/supervised/data.py`: the manifest stays a GeoDataFrame from
  `table.read`. Rows become Items through `table.to_items` wherever an asset
  or property is read; no cell is read by hand, and stac-geoparquet's
  `to_dict` import goes.
- `workflow/tasks/labels.py`: unchanged calls, `table.from_items` and
  `table.read`.
- `docs/guides/`: STAC sections follow the caller view above.

## Breaking changes

- Assets publish `raster:bands` / `eo:bands` instead of 1.1 `bands`; `cf:` and
  `cube:` fields are no longer written.
- `table.load` takes Items, not rows; call `table.to_items(rows)` first.
- A table is one file; a folder of part files is no longer read as one.
- `StacSourceConfig.asset_fields` uses v1.1.0 band keys.

## Tests

Tests mirror the source: `tests/geodata/attrs/headers/test_stac.py`,
`tests/geodata/stac/extensions/test_{zarr,geosave}.py`,
`tests/geodata/stac/test_{asset,item,table,table_client}.py`. The tests of the
deleted modules are deleted.

- Each extension module: what it writes through its PySTAC class, and that
  it writes and declares nothing when the raster lacks its facts.
- Each module's `read` returns the models its band states, and nothing for a
  bare band.
- `read_bands` and `band_attrs` read legacy listings, 1.1 `bands`, and
  Parquet-nulled bands into the same attrs.
- Round trip: raster -> `to_cog` / `to_zarr` -> Items -> `table.write` ->
  `StacTableClient.search` -> `table.load` gives the same variables, attrs
  and stored values.
- A stack round-trips as one collection per group and regroups by
  `geosave:stack`.
- A table moved with its assets still loads.
- A batch concatenated onto a read table and written again reads back whole.
- Building Items computes no pixels.
- `Item.validate()` on built Items, marked `integration` because it fetches
  schemas.

## End-to-end spike

Run 2026-10-08 as a throwaway script on `build_raster` data: a packed two-date
optical raster and a coloured label, written with `to_cog`, described with the
PySTAC classes above, stored and found through rustac, and opened with `read_raster`.

- Each optical Item declared Projection, Raster and EO; the label Item
  declared Projection, Raster and Classification, and no EO.
- Scale, nodata, dtype, unit and classes read back through the typed classes.
- The loaded Dataset had both dates, both variables, `uint16` and its attrs.
- Items validated once the collection id without a link was removed.
- `gs.timespan` widens a midnight label to its whole day, which is why a
  single label uses `gs.times` for `datetime` instead.

## Risks

- The `geosave` schema URI resolves only once `schemas/` is pushed to `main`.
  A stack Item validates against the local file (tested).
- A float32 `scale_factor` is published as its widened float64 value, as the
  attrs model reads it.
- The collection link is stored as an absolute href to the table, so it goes
  stale when a catalog moves; Collections are read from the file itself, not
  through the link.
- Rewriting a table without `collections=` drops the stored Collections.
- A fill read back from a COG is a float, so a `uint16` band publishes
  `"nodata": 0.0`.
- A store's Item now ends at the end of its last date (`23:59:59.999999`),
  as `gs.timespan` reads a date label, where it ended at midnight before.

## Out of scope

- Tile extension (raster -> tile -> chip); its own spec, building on
  `GeosaveExtension`.
- A class for STAC 1.1 `bands`.
- CF and Datacube fields.
