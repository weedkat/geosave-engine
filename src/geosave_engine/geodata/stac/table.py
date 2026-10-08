"""Keep STAC Items as a table: in memory, on disk, and back to pixels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pystac
import rustac
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

from geosave_engine.geodata.io import geoparquet, storage
from geosave_engine.geodata.io.readers import read_raster, read_vector
from geosave_engine.geodata.io.storage import absolute_location

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset, GeoDataFrame
    from geosave_engine.geodata.io.storage import StorageOptions


def from_items(items: Iterable[pystac.Item]) -> GeoDataFrame:
    """Build a table with one row per Item.

    Args:
        items: Items whose assets carry absolute hrefs, or a self href that
            resolves relative ones. They are not changed.

    Returns:
        GeoDataFrame in WGS84 with the Items' properties as columns and their
        assets as mappings, every href absolute where its Item could resolve
        it.

    Raises:
        ValueError: No Item is given.

    Examples:
        >>> from_items(create_items(paths))["id"].tolist()
        ['forest_20250601T000000', 'forest_20250611T000000']
    """
    resolved = []
    for item in items:
        # A row keeps no self href, so a relative href is resolved while it can be.
        if item.get_self_href() is not None:
            item = item.clone()
            item.make_asset_hrefs_absolute()
        resolved.append(item)
    return _rows(resolved)


def to_items(rows: gpd.GeoDataFrame) -> list[pystac.Item]:
    """Build the Item each row of an item table states.

    Args:
        rows: Rows of an item table, as `from_items` or `read` returns them,
            filtered or concatenated like any GeoDataFrame.

    Returns:
        One Item per row, in row order, holding only the assets its row has.

    Examples:
        >>> [item.id for item in to_items(rows[rows["collection"] == "label"])]
        ['s0/label']
    """
    found = rustac.from_arrow(_arrow(rows))
    return [pystac.Item.from_dict(fields) for fields in found["features"]]


def write(
    items: Iterable[pystac.Item] | gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    collections: Iterable[pystac.Collection] | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> Path | str:
    """Write Items, or an item table, as one stac-geoparquet file.

    Asset hrefs are stored relative to the table, so a table and its assets
    move together. Add a batch by concatenating its rows onto the table's and
    writing again, passing the Collections the table already holds.

    Args:
        items: Items to store, or a table of them as `read` returns it.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        collections: Collections the rows belong to, stored in the file's
            metadata with the extent their rows cover. The objects are not
            changed. None stores none.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Local writes return a path; URL writes return the supplied URL.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: No Item is given, the suffix is wrong, or `collections`
            is given and a row names a collection absent from it.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> items = create_items(paths, collection=forest)
        >>> write(items, "data/catalog.parquet", collections=[forest])
        PosixPath('data/catalog.parquet')
    """
    location = absolute_location(path)
    given = to_items(items) if isinstance(items, gpd.GeoDataFrame) else list(items)

    described: dict[str, pystac.Collection] = {}
    for collection in collections or ():
        clone = collection.clone()
        # A Collection stored in a table has no document of its own to link.
        clone.clear_links()
        clone.set_self_href(location)
        described[clone.id] = clone

    stored = []
    for item in given:
        clone = item.clone()
        clone.set_self_href(location)
        clone.make_asset_hrefs_relative()
        if described and clone.collection_id is not None:
            if clone.collection_id not in described:
                raise ValueError(
                    f"item {clone.id!r} names the collection "
                    f"{clone.collection_id!r}, which is not among the collections "
                    f"given {sorted(described)}; pass it too"
                )
            # The Item schema requires a link beside a collection id.
            clone.set_collection(described[clone.collection_id])
        stored.append(clone)

    table = _arrow(_rows(stored))
    fields = {}
    for collection_id, collection in described.items():
        members = [item for item in stored if item.collection_id == collection_id]
        if members:
            collection.extent = pystac.Extent.from_items(members)
        collection.clear_links()
        fields[collection_id] = collection.to_dict(include_self_link=False)

    return storage.write(
        path,
        lambda target: to_parquet(table, target, collections=fields or None),
        suffixes=geoparquet.FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )


def read(path: str | PathLike[str], **options: Any) -> GeoDataFrame:
    """Read an item table.

    Args:
        path: Parquet file, as a path or URL.
        **options: GeoParquet read options. `bbox` and `filters` are applied
            while reading.

    Returns:
        GeoDataFrame whose asset hrefs open directly, each row holding only
        the assets it has. A GeoParquet file that is not an item table is
        returned as read.

    Examples:
        >>> rows = read("data/catalog.parquet", bbox=(12.0, 45.0, 13.0, 46.0))
        >>> rows.gs.query(anchor)["id"].tolist()
        ['forest_20250601T000000']
    """
    frame = read_vector(path, **options)
    # A table stating no STAC version is a caller's own, and stays as written.
    if "stac_version" not in frame or frame.empty:
        return frame
    location = absolute_location(path)
    items = to_items(frame)
    for item in items:
        item.set_self_href(location)
        item.make_asset_hrefs_absolute()
    return from_items(items)


def read_collections(
    path: str | PathLike[str], *, storage_options: StorageOptions | None = None
) -> dict[str, pystac.Collection]:
    """Return the Collections an item table carries, keyed by id.

    Args:
        path: Parquet file, as a path or URL.
        storage_options: Options for the filesystem a URL names.

    Returns:
        {
            "<collection id>": its Collection, with the extent of its rows,
        }
        Empty where the file carries none.

    Examples:
        >>> read_collections("data/catalog.parquet")["forest"].license
        'CC-BY-4.0'
    """
    filesystem, target = storage.filesystem_path(path, storage_options)
    with filesystem.open(target, "rb") as handle:
        metadata = pq.read_metadata(handle).metadata or {}
    return parse_collections(metadata.get(b"stac-geoparquet", b"{}"))


def parse_collections(value: bytes | str) -> dict[str, pystac.Collection]:
    """Decode the Collections one file's `stac-geoparquet` metadata states.

    Args:
        value: The metadata value, as JSON.

    Returns:
        {
            "<collection id>": its Collection,
        }

    Examples:
        >>> parse_collections(b'{"version": "1.0.0"}')
        {}
    """
    stored = orjson.loads(value).get("collections", {})
    return {
        collection_id: pystac.Collection.from_dict(fields)
        for collection_id, fields in stored.items()
    }


def load(
    items: Iterable[pystac.Item],
    *,
    assets: str | Sequence[str] | None = None,
    **options: Any,
) -> Dataset:
    """Open the rasters Items point at as one lazy Dataset.

    Args:
        items: Items holding one raster: a store, or the scenes of one
            product.
        assets: Asset name or names to read. None reads every data asset.
        **options: Raster read options. Chunks default to an empty mapping.

    Returns:
        Lazy Dataset; scenes join along `time`.

    Raises:
        KeyError: A named asset is in no Item.
        ValueError: No Item has a data asset, or the files do not form one
            raster.

    Examples:
        >>> load(to_items(rows[rows["collection"] == "optical"])).sizes["time"]
        2
    """
    wanted = [assets] if isinstance(assets, str) else assets
    hrefs = []
    found = set()
    for item in items:
        for key, asset in item.assets.items():
            if wanted is not None and key not in wanted:
                continue
            if asset.roles is not None and not asset.has_role("data"):
                continue
            hrefs.append(asset.get_absolute_href())
            found.add(key)

    for key in wanted or ():
        if key not in found:
            raise KeyError(key)
    if not hrefs:
        raise ValueError("load needs at least one Item with a data asset")
    source = hrefs[0] if len(hrefs) == 1 else hrefs
    return read_raster(source, **{"chunks": {}, **options})


def _rows(items: Sequence[pystac.Item]) -> GeoDataFrame:
    """Tabulate Items as they are, hrefs untouched."""
    records = [item.to_dict(include_self_link=False) for item in items]
    if not records:
        raise ValueError("from_items needs at least one STAC Item")
    table = parse_stac_items_to_arrow(records).read_all()
    frame = gpd.GeoDataFrame.from_arrow(table)
    # geopandas drops this covering column on read, so no item table carries it.
    return cast("GeoDataFrame", frame.drop(columns="bbox"))


def _arrow(rows: gpd.GeoDataFrame) -> pa.Table:
    """Spell rows as the Arrow table the format states, bounds and all."""
    # A STAC bbox is its geometry's bounds; geopandas dropped the column on read.
    bounds = rows.geometry.bounds
    bounds.columns = ["xmin", "ymin", "xmax", "ymax"]
    boxes = pd.Series(bounds.to_dict("records"), index=rows.index)
    return pa.table(rows.assign(bbox=boxes).to_arrow())
