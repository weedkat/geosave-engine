"""Build native STAC Items and Collections from saved rasters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC
from datetime import datetime as DateTime
from os import PathLike
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from shapely import union_all
from shapely.geometry import mapping

from geosave_engine.geodata.io.readers import read_raster
from geosave_engine.geodata.io.storage import absolute_location

from . import extensions
from .extensions import GeosaveExtension

type ItemTime = DateTime | tuple[DateTime, DateTime]
type Saved = Mapping[str, xr.Dataset]

# What STAC calls a saved raster, by its suffix.
_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".zarr": pystac.MediaType.ZARR,
    ".nc": pystac.MediaType.NETCDF,
    ".nc4": pystac.MediaType.NETCDF,
    ".cdf": pystac.MediaType.NETCDF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
}


def create_collection(
    id: str,
    *,
    description: str,
    license: str = "other",
    title: str | None = None,
    providers: Sequence[pystac.Provider] | None = None,
) -> pystac.Collection:
    """Build a Collection that Items are created into.

    Args:
        id: Collection identity, which its Items carry.
        description: What the Collection holds.
        license: SPDX license identifier, or `"other"`.
        title: Short human-readable name.
        providers: Organizations that produced, processed or host the data.

    Returns:
        Collection covering the whole globe over an unbounded time range,
        until a table states what its rows cover.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> create_items(paths, collection=forest)[0].collection_id
        'forest'
    """
    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
        pystac.TemporalExtent([[None, None]]),
    )
    return pystac.Collection(
        id=id,
        description=description,
        extent=extent,
        title=title,
        license=license,
        providers=None if providers is None else list(providers),
    )


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

    Returns:
        Item over the files' footprint, dated by one instant or by a start
        and end, declaring the extensions its assets use. No pixel is read.

    Raises:
        ValueError: An asset has no locatable grid or names no raster format,
            or no asset is dated and `datetime` is None.

    Examples:
        >>> create_item({"label": "s0/label.tif"}, id="s0").id
        's0'
    """
    with ExitStack() as opened:
        named = {key: _open(opened, path, options) for key, path in assets.items()}
        return _item(named, id=id, collection=_identity(collection), when=datetime)


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
        paths: What a raster's `to_cog`, `to_zarr` or `to_netcdf` returned.
        id: `GeoAnchor.format` template filled from each Item's own raster,
            as `"{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}"`. None names each Item
            after its file, or the folder its files share.
        collection: Collection the Items belong to; they take its id. None
            states none.
        datetime: Time for timeless files.
        **options: Read options passed to `read_raster` for every file.

    Returns:
        One Item per instant the files hold, in the order given, with one
        asset per file; a store is one Item. No pixel is read.

    Raises:
        ValueError: No path is given, two files of one instant claim the same
            asset key, `id` gives two Items the same id, or a file is
            timeless and `datetime` is None.

    Examples:
        >>> [item.id for item in create_items(ds.gs.to_cog("samples/forest"))]
        ['forest_20250601T103031', 'forest_20250611T103031']
    """
    files = [paths] if isinstance(paths, (str, PathLike)) else list(paths)
    if not files:
        raise ValueError("create_items needs the paths a writer returned")
    with ExitStack() as opened:
        saved = dict(_open(opened, path, options) for path in files)
        return _items(saved, id=id, collection=_identity(collection), when=datetime)


def create_stack_items(
    paths: Mapping[str, str | PathLike[str] | Sequence[str | PathLike[str]]],
    *,
    name: str,
    datetime: ItemTime | None = None,
    **options: Any,
) -> tuple[pystac.Item, ...]:
    """Build the Items of every group of a saved stack.

    Args:
        paths: What a stack's `to_cog`, `to_zarr` or `to_netcdf` returned:
            group names mapped to the files, or the one store, holding each.
        name: The stack's identity, shared by its Items.
        datetime: Time for timeless groups. None gives them the span the
            dated groups cover.
        **options: Read options passed to `read_raster` for every file.

    Returns:
        Each group's Items, groups in the order `paths` lists them. A group's
        name is its collection, ids are prefixed with `name`, and every Item
        carries `geosave:stack`.

    Raises:
        ValueError: A group is timeless, no group is dated and `datetime` is
            None.

    Examples:
        >>> items = create_stack_items(sample.gs.to_cog("samples/s0"), name="s0")
        >>> [item.id for item in items]
        ['s0/optical_20250601T103031', 's0/optical_20250611T103031', 's0/label']
        >>> items[0].collection_id
        'optical'
    """
    with ExitStack() as opened:
        groups: dict[str, Saved] = {}
        for group, saved in paths.items():
            files = [saved] if isinstance(saved, (str, PathLike)) else list(saved)
            groups[group] = dict(_open(opened, path, options) for path in files)

        # A timeless group is dated by what the dated groups cover.
        spans = [
            raster.gs.timespan
            for files in groups.values()
            for raster in files.values()
            if raster.gs.timespan is not None
        ]
        fallback = datetime
        if fallback is None and spans:
            fallback = (min(start for start, _ in spans), max(end for _, end in spans))

        items: list[pystac.Item] = []
        for group, files in groups.items():
            dated = any(raster.gs.timespan is not None for raster in files.values())
            built = _items(
                files,
                id=None,
                collection=group,
                when=None if dated else fallback,
                prefix=f"{name}/",
            )
            for each in built:
                GeosaveExtension.ext(each, add_if_missing=True).stack = name
            items.extend(built)
        return tuple(items)


def default_key(raster: xr.Dataset) -> str:
    """Name the asset holding a whole raster.

    Args:
        raster: Raster one file or store holds.

    Returns:
        The raster's variable where it has exactly one, else `"image"`.

    Examples:
        >>> default_key(scene[["red"]]), default_key(scene)
        ('red', 'image')
    """
    variables = raster.gs.variables
    if len(variables) == 1:
        return variables[0]
    return "image"


def _media_type(href: str) -> str:
    """Name the media type of the raster file or store an href points at."""
    found = _MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if found is None:
        raise ValueError(
            f"{href} names no raster file or store; pass a path a writer returned"
        )
    return found


def _identity(collection: pystac.Collection | None) -> str | None:
    """Return the id an Item of this Collection carries, None for no Collection."""
    return None if collection is None else collection.id


def _open(
    opened: ExitStack, path: str | PathLike[str], options: Mapping[str, Any]
) -> tuple[str, xr.Dataset]:
    """Open one saved raster lazily, to be closed when `opened` exits."""
    raster = opened.enter_context(read_raster(path, **options))
    return absolute_location(path), raster


def _items(
    saved: Saved,
    *,
    id: str | None,
    collection: str | None,
    when: ItemTime | None,
    prefix: str = "",
) -> tuple[pystac.Item, ...]:
    """Build one Item per scene the saved files hold.

    Args:
        saved: Absolute href mapped to the raster opened from it, files of
            one raster in the order its writer returned them.
        id: `GeoAnchor.format` template, or None for a name from the path.
        collection: Collection id the Items carry, or None.
        when: Time for timeless files.
        prefix: Text put before every id.

    Returns:
        One Item per scene, in the order the scenes first appear.

    Raises:
        ValueError: Two files of one scene claim the same asset key, or two
            Items share an id.
    """
    # Files covering the same time are one scene; a store covers its own axis.
    scenes: dict[object, dict[str, xr.Dataset]] = {}
    for href, raster in saved.items():
        scenes.setdefault(raster.gs.timespan, {})[href] = raster

    items = []
    for files in scenes.values():
        named: dict[str, tuple[str, xr.Dataset]] = {}
        for href, raster in files.items():
            key = default_key(raster)
            if key in named:
                raise ValueError(
                    f"{href} and {named[key][0]} cover the same time and would "
                    f"both be the asset {key!r}; build them as separate Items"
                )
            named[key] = (href, raster)

        # One file is named after itself; several after the folder they share.
        first_href, first = next(iter(files.items()))
        written = PurePosixPath(first_href)
        name = written.stem if len(files) == 1 else written.parent.name
        item_id = name if id is None else first.gs.anchor.format(id)
        items.append(
            _item(named, id=f"{prefix}{item_id}", collection=collection, when=when)
        )

    ids = [each.id for each in items]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"id template {id!r} gives several scenes the same id {ids}; add a "
            f"time field such as {{start:%Y%m%d}}"
        )
    return tuple(items)


def _item(
    named: Mapping[str, tuple[str, xr.Dataset]],
    *,
    id: str,
    collection: str | None,
    when: ItemTime | None,
) -> pystac.Item:
    """Build one Item from opened rasters keyed by asset name.

    Args:
        named: Asset key mapped to the href and the raster opened from it.
        id: Item identity.
        collection: Collection id the Item carries, or None.
        when: Time for timeless rasters.

    Returns:
        Item with one asset per raster, over their joined footprint.

    Raises:
        ValueError: A raster has no locatable grid, or none is dated and
            `when` is None.
    """
    rasters = [raster for _, raster in named.values()]
    if isinstance(when, tuple):
        start, end = when
    elif when is not None:
        start, end = when, when
    else:
        start, end = _covered(rasters, list(named))

    grids = [raster.gs.geobox for raster in rasters]
    if any(grid is None or grid.crs is None for grid in grids):
        footprint = None
    else:
        footprint = union_all([grid.geographic_extent.geom for grid in grids])

    # STAC dates an Item by one instant, or by a start and an end, never both.
    ranged = start != end
    item = pystac.Item(
        id,
        None if footprint is None else mapping(footprint),
        None if footprint is None else list(footprint.bounds),
        None if ranged else start,
        {},
        start_datetime=start if ranged else None,
        end_datetime=end if ranged else None,
        collection=collection,
    )
    for key, (href, raster) in named.items():
        # An extension declares its schema on the asset's owner, so add first.
        item.add_asset(
            key, pystac.Asset(href, media_type=_media_type(href), roles=["data"])
        )
        for extension in extensions.ASSET:
            extension.write(item.assets[key], raster)
    return item


def _covered(
    rasters: Sequence[xr.Dataset], keys: Sequence[str]
) -> tuple[DateTime, DateTime]:
    """Return the first and last instant the rasters of one Item cover.

    Args:
        rasters: Rasters of one Item, as opened from their files.
        keys: Their asset keys, for the error.

    Returns:
        (first, last) in UTC. Rasters holding one time label between them
        cover that instant; rasters holding several cover what `gs.timespan`
        says, so a monthly label covers its month.

    Raises:
        ValueError: No raster is dated.
    """
    labels = [raster.gs.times for raster in rasters if raster.gs.times is not None]
    spans = [raster.gs.timespan for raster in rasters if raster.gs.timespan is not None]
    if not spans:
        raise ValueError(
            f"none of the assets {keys} is dated, and a STAC Item needs a "
            f"time; pass datetime="
        )
    instants = pd.DatetimeIndex(
        np.concatenate([each.values for each in labels])
    ).unique()
    if len(instants) == 1:
        instant = instants[0].tz_localize("UTC").to_pydatetime()
        return instant, instant
    start = min(start for start, _ in spans).replace(tzinfo=UTC)
    end = max(end for _, end in spans).replace(tzinfo=UTC)
    return start, end
