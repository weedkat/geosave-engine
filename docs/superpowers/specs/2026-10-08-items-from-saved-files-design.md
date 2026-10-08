# Items From Saved Files Design

**Status:** Implemented 2026-10-08. Supersedes the object-based builders

> **Later change, 2026-10-08:** `stac/band.py` and `stac/header.py` no longer
> exist. Band encoding is `stac.asset.band_fields`; band reading and decoding
> (`read_bands`, `band_attrs`) sit with `create_header` in
> `attrs/headers/stac.py`. `StacTableClient` lives in `stac/client.py`.
of `2026-10-06-items-from-rasters-design.md`.

**Goal:** Build STAC Items from the files a writer saved, so an Item has one
source of truth and `stac/` needs no layout guessing.

## Cause being removed

Today an Item is built from two inputs that must agree and are not linked: the
in-memory raster and a bare list of paths.

```python
paths = ds.gs.to_cog("samples/forest")   # the writer knows scene, variable, instant per file
items = ds.gs.to_items(paths)            # item.py works that out again from path strings
```

That one fact produces `_from_cogs` (`divmod`, suffix and folder-name
guessing), the "raster must be the one saved there" contract, the
`{**attrs, **encoding}` rule in `band.py`, and tests that only check the two
descriptions agree. Probed 2026-10-08: the object says `uint16` where a
writer-level `encoding` stored `uint8`; a COG drops a zero offset and adds a
description the object lacks.

## Decision

The saved file is the source. Each file is opened once, lazily, and everything
an Item states comes from that header. Measured locally: 9.6 ms per COG through
`read_raster`, 29 ms for a Zarr store, 12 ms for NetCDF.

## Caller view

```python
from geosave_engine.geodata import stac

forest = stac.create_collection("forest", description="Forest samples", license="CC-BY-4.0")

paths = ds.gs.to_cog("samples/forest", split_bands=True)
items = stac.create_items(paths, collection=forest)
stac.table.write(items, "samples/catalog/part-0.parquet", collections=[forest])

# Later: more Items into the same Collection are another part file.
more = stac.create_items(new_paths, collection=forest)
stac.table.write(more, "samples/catalog/part-1.parquet", collections=[forest])

rows = stac.table.read("samples/items.parquet", bbox=(12.0, 45.0, 13.0, 46.0))
ds = stac.table.load(rows)
collections = stac.table.read_collections("samples/items.parquet")
```

One Item with caller-named assets, and a saved stack:

```python
item = stac.create_item({"label": "s0/label.tif"}, id="s0")
items = stac.create_stack_items(sample.gs.to_cog("samples/s0"), name="s0")
```

## What an Item and an asset are

An asset is one thing `read_raster` opens by its href. An Item is the assets
that hold one instant, or one store spanning its instants.

| Saved as | Items | Assets per Item | Asset key |
| --- | --- | --- | --- |
| COG, bands together | one per instant | one multi-band file | the variable, or `image` for several |
| COG, `split_bands=True` | one per instant | one file per variable | the variable |
| Zarr or NetCDF | one for the store | the store | the variable, or `image` for several |
| Stack, COG | each group's Items, collection = group | as above | as above |
| Stack, Zarr or NetCDF | one per group, collection = group | that group's own store | as above |

## Public API

```python
# stac/item.py
type ItemTime = DateTime | tuple[DateTime, DateTime]


def create_item(
    assets: Mapping[str, str | PathLike[str]],
    *,
    id: str,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> pystac.Item:
    """Build one Item from saved rasters the caller names.

    Args:
        assets: Asset key mapped to the file or store holding it, as
            `{"label": "s0/label.tif"}`.
        id: Item identity.
        collection: Collection the Item belongs to; the Item takes its id.
            None states none.
        datetime: Time for timeless files, as one instant or a start and end.
            None takes the instants of the files' `time` coordinate, widened
            to what a `TimeSpec` on it says each label covers.
        **options: Read options passed to `read_raster` for every asset.

    Raises:
        ValueError: An asset has no locatable grid or no raster suffix, or no
            asset is dated and `datetime` is None.
    """


def create_items(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    id: str | None = None,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> tuple[pystac.Item, ...]:
    """Build the Items of one saved raster.

    Args:
        paths: What `to_cog`, `to_zarr` or `to_netcdf` returned.
        id: `GeoAnchor.format` template filled from each Item's own raster,
            as `"{lat:.2f}N_{start:%Y%m%d}"`. None names each Item after the
            file, or the folder its files share.

    Returns:
        One Item per instant the files hold, in time order; a store is one
        Item.

    Raises:
        ValueError: No path is given, two files of one instant claim the same
            asset key, or `id` gives two Items the same id.

    Examples:
        >>> [item.id for item in create_items(ds.gs.to_cog("samples/forest"))]
        ['forest_20250601T103031', 'forest_20250611T103031']
    """


def create_stack_items(
    paths: str | PathLike[str] | Mapping[str, Sequence[str | PathLike[str]]],
    *,
    name: str,
    datetime: ItemTime | None = None,
    **options: Any,
) -> tuple[pystac.Item, ...]:
    """Build the Items of every group of a saved stack.

    Returns:
        Each group's Items in group order. The group is their collection,
        ids are prefixed with `name`, and each carries `geosave:stack`. A
        timeless group takes the span the dated groups cover.
    """


def create_collection(
    id: str,
    *,
    description: str,
    license: str = "other",
    title: str | None = None,
    providers: Sequence[pystac.Provider] | None = None,
) -> pystac.Collection:
    """Build a Collection that Items are created into.

    Returns:
        Collection with an open extent: the whole globe and an unbounded
        time range, until a table states what its rows cover.
    """
```

`stac/__init__.py` exports the four `create_*` functions beside `table`.

```python
# stac/table.py
def write(
    items: Iterable[pystac.Item] | gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    collections: Sequence[pystac.Collection] | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> Path | str: ...


def read_collections(
    path: str | PathLike[str], *, storage_options: StorageOptions | None = None
) -> dict[str, pystac.Collection]:
    """Return the Collections a table's files carry, keyed by id.

    Examples:
        >>> read_collections("samples/items.parquet")["forest"].license
        'CC-BY-4.0'
    """
```

`from_items`, `read` and `load` keep their signatures.

## Internal structure

```text
stac/
├── band.py     variable -> band, band -> attrs, asset -> bands
├── asset.py    from_raster(raster, href) -> pystac.Asset        (opens nothing)
├── item.py     create_item, create_items, create_stack_items, create_collection
└── table.py    from_items, write, read, read_collections, load
```

- **`item.py` is the only place a file is opened.** `create_item` opens each
  path once with `read_raster`; that one raster yields the asset
  (`asset.from_raster`), the footprint, the time and the id fields, then closes.
- **`create_items` groups, then delegates.** It reads each file's instant from
  its header, groups files by instant, keys each by `asset.default_key`, and
  calls `create_item` per group. A store is a group of one.
- **The footprint is the union of the assets' extents**, so assets on
  different grids are allowed.
- **Time is read once.** The Item's time comes from the opened rasters; nothing
  is written to an asset and parsed back. Assets carry no `start_datetime`.
- **`asset.from_raster` loses `_describe`.** One flat body: grid guard, media
  type from the suffix, extension fields, `bands`.
- **`band.from_variable` reads `.attrs` only.** Files are opened raw, so the
  `.encoding` merge goes.
- **`table.load` selects, then opens.**

```python
class DataAsset(NamedTuple):
    """One data asset of a selected row."""

    key: str                    # 'red'
    href: str                   # '/data/forest/forest_20250601T103031/red.tif'


def _data_assets(rows: gpd.GeoDataFrame, wanted: Sequence[str] | None) -> list[DataAsset]: ...
```

  Every asset opens by its href alone. A stack is saved as one store per
  group (`plans/2026-10-08-stack-store-per-group.md`), so no asset names a
  group inside a store and `xarray:open_kwargs` is neither written nor read.

## Collections in the table

- A Collection exists on its own. `create_items(..., collection=forest)` copies
  `forest.id` onto each Item; the id is never typed twice. PySTAC's
  `set_collection` and `add_items` are not used: probed 2026-10-08, they write
  links with `href: null` into every row.
- `write(..., collections=[...])` passes `{id: collection.to_dict()}` to
  stac-geoparquet, which stores it in the file metadata under
  `stac-geoparquet.collections`. Rows carry only the id.
- The extent written is computed from the rows naming that Collection: their
  bounds, and their first and last instant. The object passed in is unchanged.
- One table may hold several Collections; a stack's groups rely on that.
- When `collections` is given, an Item naming a collection absent from it
  raises `ValueError`. When it is None, none is written.
- `read_collections` reads one file, or every part file of a folder. For one
  id it unions the extents; any other field described differently in two
  files raises `ValueError`.
- Reading a table, editing rows and writing it back keeps Collections only if
  they are passed again.

## Removed

- `item.from_raster`, `item.from_files`, `item._from_cogs`, `item._span`,
  `item.from_stack`.
- `ds.gs.to_items`, `array.gs.to_items`, `stack.gs.to_items`: the object is no
  longer an input.
- The `pystac.Asset` branch of `asset.default_key`, and asset-level
  `start_datetime` / `end_datetime`.
- The `.encoding` rule in `band.from_variable`.
- Tests asserting an asset built at write time agrees with the file.

`workflow/tasks/labels.py` calls `create_item({"label": path}, id=sample_id)`;
`ml/segmentation/supervised/data.py` is unchanged.

## Limits

- A remote file costs one header request and needs its storage options in
  `**options`.
- A deferred write (`compute=False`) is described after it has run.
- The default id is taken from the path; every other fact comes from headers.
- Group order follows the mapping a stack writer returned. A folder of
  groups reads in name order.
- A COG names an undescribed band after itself and stores no zero offset, so
  those two fields read back differently from the raster that was written.

## Tests

- Each row of the Item/asset table above, for COG, Zarr and NetCDF.
- A writer-level `encoding` override is described as stored.
- `create_item`: caller keys, timeless without `datetime`.
- `create_stack_items`: COG files and one store per group; a timeless label takes
  the optical span; `table.load` of each group opens that group.
- Collections: two in one table round trip; dangling id raises; conflicting
  part files raise.
- `table.load`: conflicting open options raise.
- Laziness: building Items computes no pixels.
- The existing attrs -> Item -> GeoParquet -> attrs round trip keeps passing.

## Follow-up spec

`StacSource.load(anchor, return_stac=True)` returning the source Items and
Collection; `create_items(..., derived_from=)` writing `derived_from` links;
whether captured source properties are copied onto new Items.
