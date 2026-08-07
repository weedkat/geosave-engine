"""Free-function geometry ops on GeoTile: remap, align_spatial, split_spatial, mosaic_spatial/mosaic_stack, chunk_geotile.

Not GeoTile methods — keeps GeoTile's own surface to its own fields/serialization.
Disk I/O and shape validation live in geodata.utils.io/.geodata instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, TypeVar, cast

import geopandas as gpd
import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray/Dataset
import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.geom import Geometry
from rasterio.enums import Resampling
from rioxarray.merge import merge_datasets

from geosave_engine.geodata.utils.geodata import da_to_ds, ds_to_da

from .geoanchor import GeoAnchor, GeoTag
from .geotile import GeoTile

if TYPE_CHECKING:
    from .geostack import GeoStack

MergeMethod = Literal["first", "last", "min", "max", "sum", "count"] | Callable
# time-coord floor precision for merging; GeoTag datetime stays exact either way
DatePrecision = Literal["D", "W", "M", "Y"] | None
T = TypeVar("T", bound=GeoAnchor)


def _floor_time(da: xr.DataArray, precision: DatePrecision) -> xr.DataArray:
    """Floor da's time coord to precision's bucket start. No-op if precision is None or da has no time dim."""
    if precision is None or "time" not in da.dims:
        return da
    if precision in ("D", "W"):
        return da.assign_coords(time=da.time.dt.floor(precision))
    floored = pd.DatetimeIndex(da.time.values).to_period(precision).to_timestamp()
    return da.assign_coords(time=("time", floored))


def remap(tile: GeoTile, mapping: dict[int, int]) -> GeoTile:
    """Return a new GeoTile with label values remapped per ``mapping``."""
    remapped = tile.data
    for src_val, dst_val in mapping.items():
        remapped = remapped.where(remapped != src_val, other=dst_val)
    return tile.rebase(data=remapped)


def align_spatial(*tiles: GeoTile, tol: float = 1e-6) -> tuple[GeoTile, ...]:
    """Narrow each tile's geobox to their common intersection.

    Pure geometry — data is shared untouched.

    Args:
        *tiles: Tiles to align, same CRS/resolution/pixel grid.
        tol: Floating-point tolerance for resolution and pixel-grid matching.

    Returns:
        Same count as input, each narrowed to the shared intersection.
        Fewer than 2 tiles: input returned as-is, nothing to narrow.

    Raises:
        ValueError: CRS not projected, mixed CRS/resolution, tiles don't
            overlap, or not on a common pixel grid.
    """
    if len(tiles) < 2:
        return tiles  # nothing to align, just return the single tile

    # CRS must be projected — resolution only means meters under one
    crs = tiles[0].geobox.crs
    if crs is None or not crs.projected:
        raise ValueError(f"align_spatial() needs a projected CRS, got {tiles[0].crs}")

    # check each tile against the one before it: CRS/resolution match, bbox still overlaps
    minx, miny, maxx, maxy = tiles[0].bbox
    for i, t in enumerate(tiles[1:], start=1):
        prev = tiles[i - 1]
        if t.crs != prev.crs:
            raise ValueError(f"align_spatial(): tile {i} has different CRS than tile {i - 1}")
        if abs(t.resolution - prev.resolution) > tol:
            raise ValueError(f"align_spatial(): tile {i} has different resolution than tile {i - 1}")
        minx, miny = max(minx, t.bbox[0]), max(miny, t.bbox[1])
        maxx, maxy = min(maxx, t.bbox[2]), min(maxy, t.bbox[3])
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"align_spatial(): tile {i} doesn't overlap the rest")

    # narrow each tile's geobox to that shared bbox, clip its polygon to match
    aligned: list[GeoTile] = []
    for i, t in enumerate(tiles):
        res = t.resolution
        left, _, _, top = t.bbox
        col0 = (minx - left) / res
        row0 = (top - maxy) / res
        ncols = (maxx - minx) / res
        nrows = (maxy - miny) / res
        if any(abs(v - round(v)) > tol for v in (col0, row0, ncols, nrows)):
            raise ValueError(f"align_spatial(): tile {i} not on the common pixel grid")
        col0, row0, ncols, nrows = round(col0), round(row0), round(ncols), round(nrows)
        sub = t.geobox[row0:row0 + nrows, col0:col0 + ncols]
        aligned_t = t.rebase(geobox=sub)
        if t.polygon is not None:
            clip_box = Geometry(
                {
                    "type": "Polygon",
                    "coordinates": [[(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]],
                },
                crs=t.polygon.crs,
            )
            aligned_t = aligned_t.rebase(polygon=t.polygon & clip_box)
        aligned.append(aligned_t)
    return tuple(aligned)


def split_spatial(*tiles: GeoTile) -> list[tuple[GeoTile, ...]]:
    """Group tiles into spatially-connected clusters — touching/overlapping tiles end up together.

    Pure discovery, no merging — footprints reproject to one CRS for the
    connectivity check only; each tile's own data/CRS stays untouched.

    Args:
        *tiles: Tiles to group, any CRS.

    Returns:
        One tuple per connected cluster, original relative order kept.
        A tile touching nothing else comes back as its own singleton group.

    Raises:
        ValueError: tiles is empty, or tile 0 has no CRS.
    """
    if not tiles:
        raise ValueError("split_spatial() requires at least one tile")
    if len(tiles) == 1:
        return [tiles]

    # every footprint reprojected onto tile 0's CRS, just for the connectivity check
    crs = tiles[0].crs
    if crs is None:
        raise ValueError("split_spatial() needs tile 0 to have a real CRS")
    footprints = [t.bbox_polygon.to_crs(crs).geom for t in tiles]

    # union merges touching footprints; explode splits back into disjoint pieces
    merged = gpd.GeoSeries(footprints, crs=crs).union_all()
    cluster_geoms = list(gpd.GeoSeries([merged], crs=crs).explode(index_parts=False))

    # group each tile into the first cluster it touches, preserving input order
    groups: dict[int, list[int]] = {}
    for i, fp in enumerate(footprints):
        for cluster_id, geom in enumerate(cluster_geoms):
            if geom.intersects(fp):
                groups.setdefault(cluster_id, []).append(i)
                break

    return [tuple(tiles[i] for i in idxs) for idxs in groups.values()]


def mosaic_spatial(
    *tiles: GeoTile,
    method: MergeMethod = "first",
    target_crs: str | None = None,
    target_resolution: float | None = None,
    resampling: Resampling = Resampling.nearest,
    date_precision: DatePrecision = 'D',
) -> GeoTile:
    """Merge tiles into one, gaps filled with nodata.

    No time requirement — tiles don't need to agree on time, grouping by
    time is the caller's job. GeoTag datetime stays the exact min/max
    across inputs regardless of date_precision.

    Args:
        *tiles: Tiles to merge.
        method: Overlap-resolution rule forwarded to merge_datasets.
        target_crs: Reproject mismatched tiles onto this CRS before
            merging. None requires tiles already share one CRS.
        target_resolution: Resample mismatched tiles onto this resolution
            before merging. None requires tiles already share one resolution.
        resampling: Resampling used for target_crs/target_resolution reconciliation.
        date_precision: Floor each tile's array time coord to this before
            merging. None leaves it untouched.

    Returns:
        One GeoTile spanning every input's combined footprint.

    Raises:
        ValueError: tiles is empty, CRS mismatch with target_crs unset,
            resolution mismatch with target_resolution unset, or bands don't match.
    """
    if not tiles:
        raise ValueError("mosaic_spatial() requires at least one tile")

    # CRS: target_crs given wins, else every tile must already agree on one
    if target_crs is None:
        by_crs: dict[str, list[int]] = {}
        for i, t in enumerate(tiles):
            by_crs.setdefault(str(t.crs), []).append(i)
        if len(by_crs) > 1:
            raise ValueError(f"mosaic_spatial(): mixed CRS — {by_crs} — pass target_crs=")
        target_crs = tiles[0].crs
        if target_crs is None:
            raise ValueError("mosaic_spatial() needs tiles to have a real CRS")

    # resolution: target_resolution given wins, else every tile must already agree on one
    if target_resolution is None:
        by_res: dict[float, list[int]] = {}
        for i, t in enumerate(tiles):
            by_res.setdefault(round(t.resolution, 6), []).append(i)
        if len(by_res) > 1:
            raise ValueError(f"mosaic_spatial(): mixed resolution — {by_res} — pass target_resolution=")
        target_resolution = tiles[0].resolution

    # reproject only the tiles that actually need it, one combined pass per tile
    reconciled = []
    for t in tiles:
        need_crs = str(t.crs) != target_crs
        need_res = round(t.resolution, 6) != round(target_resolution, 6)
        if need_crs or need_res:
            t = t.reproject(
                crs=target_crs if need_crs else None,
                resolution=target_resolution if need_res else None,
                resampling=resampling,
            )
        reconciled.append(t)
    tiles = tuple(reconciled)

    crs = tiles[0].geobox.crs
    if crs is None or not crs.projected:
        raise ValueError(f"mosaic_spatial() needs a projected CRS, got {tiles[0].crs}")

    # every tile needs the same band structure — no reconciliation for this
    has_band = "band" in tiles[0].data.dims
    for i, t in enumerate(tiles[1:], start=1):
        if ("band" in t.data.dims) != has_band:
            raise ValueError(f"mosaic_spatial(): tile {i} band dim differs from tile 0")
        if has_band and t.bands != tiles[0].bands:
            raise ValueError(f"mosaic_spatial(): tile {i} bands {t.bands} != tile 0 bands {tiles[0].bands}")

    # floor time coord to date_precision before merge
    datasets = [da_to_ds(_floor_time(t.data, date_precision)) for t in tiles]
    merged_ds = merge_datasets(datasets, method=method).assign_attrs(has_band=has_band)  # attrs don't carry through
    merged_da = ds_to_da(merged_ds)

    geobox = GeoBox.from_bbox(merged_da.rio.bounds(), crs=target_crs, resolution=target_resolution)

    # union polygon, skip tiles without one
    polygon: Geometry | None = None
    tile_polys = [t.polygon.to_crs(target_crs) for t in tiles if t.polygon is not None]
    if tile_polys:
        polygon = tile_polys[0]
        for p in tile_polys[1:]:
            polygon = polygon | p

    tag = GeoTag(
        datetime=(min(t.start for t in tiles), max(t.end for t in tiles)),
        metadata={k: v for t in tiles for k, v in t.metadata.items()},
        polygon=polygon,
        plot_meta=tiles[0].plot_meta,
    )
    merged_tile = GeoTile(geobox=geobox, data=merged_da, geotag=tag)
    return merged_tile.rebase(stac=[item for t in tiles for item in t.stac])


def mosaic_stack(*stacks: GeoStack, method: MergeMethod = "first") -> GeoStack:
    """Merge stacks into one, layer by layer, gaps filled with nodata.

    Layers don't need to match across stacks — each layer name merges
    across whichever stacks carry it.

    Args:
        *stacks: Stacks to merge.
        method: Forwarded to mosaic_spatial per layer.

    Returns:
        One GeoStack with every layer name found across inputs.

    Raises:
        ValueError: stacks is empty, or a layer's tiles fail mosaic_spatial's own checks.

    Examples:
        >>> region = mosaic_stack(stack_a, stack_b)  # stack_b missing a layer stack_a has is fine
    """

    from .geostack import GeoStack  # noqa: F811 — deferred to dodge the geostack.py <-> ops.py cycle

    if not stacks:
        raise ValueError("mosaic_stack() requires at least one stack")
    layer_names = {name for s in stacks for name in s.tiles}
    merged = {
        name: mosaic_spatial(*(s.tiles[name] for s in stacks if name in s.tiles), method=method)
        for name in layer_names
    }
    return GeoStack(**merged)


def chunk_geotile(full: T, tile_size_px: int) -> list[T]:
    """Chunk a GeoAnchor/GeoTile's geobox into a grid of sub-anchors.

    Args:
        full: GeoAnchor or GeoTile covering the whole area to split.
        tile_size_px: Sub-tile side length in pixels (square).

    Returns:
        One instance per grid cell intersecting full's extent, same type as full.
        full.polygon, if set, is clipped to each cell's own bbox instead of dropped.
    """
    grid = GeoboxTiles(full.geobox, (tile_size_px, tile_size_px))
    # grid[idx] types as the wider GeoBoxBase but is always concretely GeoBox here
    sub_tiles = [full.rebase(geobox=cast(GeoBox, grid[idx])) for idx in grid.tiles(full.geobox.extent)]
    polygon = full.polygon
    if polygon is None:
        return sub_tiles

    def _clip_to_cell(t: T) -> T:
        minx, miny, maxx, maxy = t.bbox
        clip_box = Geometry(
            {
                "type": "Polygon",
                "coordinates": [[(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]],
            },
            crs=polygon.crs,
        )
        return t.rebase(polygon=t.polygon & clip_box) #type: ignore

    return [_clip_to_cell(t) for t in sub_tiles]
