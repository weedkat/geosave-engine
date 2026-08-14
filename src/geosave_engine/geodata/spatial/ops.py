"""Free-function geometry ops on GeoTile: remap, align_spatial, split_spatial, mosaic_spatial/mosaic_stack, chunk_geotile.

Not GeoTile methods — keeps GeoTile's own surface to its own fields/serialization.
Disk I/O and shape validation live in geodata.utils.zarr/.geotiff/.geodata instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, cast

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray/Dataset
import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr
from dask.base import compute as dask_compute
from dask.base import is_dask_collection
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.geom import Geometry
from rasterio.enums import Resampling

from .anchor import GeoAnchor, GeoTag
from .tile import GeoTile

if TYPE_CHECKING:
    from .stack import GeoStack

MergeMethod = Literal["first", "last"]
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


def _mask_nodata(da: xr.DataArray) -> xr.DataArray:
    """da with its own nodata pixels turned into real NaN, for combine_first to skip.

    Args:
        da: Array to mask.

    Returns:
        da unchanged if it has no declared nodata or its nodata already is NaN.
    """
    nodata = da.rio.nodata
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        return da
    return da.where(da != nodata)


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
        # data untouched — narrowed geobox is enough, to_numpy()/to_tensor()'s own
        # rio.clip_box(*bbox) narrows the pixels lazily at render time
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


def split_spatial(*tiles: GeoTile, tol: float = 0.0) -> list[tuple[GeoTile, ...]]:
    """Group tiles into spatially-connected clusters — touching/overlapping tiles end up together.

    Pure discovery, no merging — footprints reproject to one CRS for the
    connectivity check only; each tile's own data/CRS stays untouched.

    Args:
        *tiles: Tiles to group, any CRS.
        tol: Max gap (CRS units, e.g. meters) between two footprints for
            them to still count as connected. 0 means touching/overlapping only.

    Returns:
        One tuple per connected cluster, original relative order kept.
        A tile touching nothing else comes back as its own singleton group.

    Raises:
        ValueError: tiles is empty, tile 0 has no CRS, or tol is negative.
    """
    if not tiles:
        raise ValueError("split_spatial() requires at least one tile")
    if tol < 0:
        raise ValueError(f"split_spatial(): tol must be >= 0, got {tol}")
    if len(tiles) == 1:
        return [tiles]

    # every footprint reprojected onto tile 0's CRS, just for the connectivity check
    crs = tiles[0].crs
    if crs is None:
        raise ValueError("split_spatial() needs tile 0 to have a real CRS")
    footprints = [t.bbox_polygon.to_crs(crs).geom for t in tiles]

    # buffer by tol/2 each so a <=tol gap between two footprints starts overlapping
    grow = footprints if tol == 0 else [fp.buffer(tol / 2) for fp in footprints]

    # union merges touching/grown footprints; explode splits back into disjoint pieces
    merged = gpd.GeoSeries(grow, crs=crs).union_all()
    cluster_geoms = list(gpd.GeoSeries([merged], crs=crs).explode(index_parts=False))

    # group each tile into the first cluster its (grown) footprint touches, preserving input order
    groups: dict[int, list[int]] = {}
    for i, fp in enumerate(grow):
        for cluster_id, geom in enumerate(cluster_geoms):
            if geom.intersects(fp):
                groups.setdefault(cluster_id, []).append(i)
                break

    return [tuple(tiles[i] for i in idxs) for idxs in groups.values()]


def align_temporal(tile: GeoTile) -> GeoTile:
    """Crop tile to the pixel bbox where every time+band step has real data.

    Border trim only — a nodata hole inside the bbox on one date stays
    nodata. No-op if tile has no time dim. Data stays lazy, never fully
    materialized: only the reduced (y,)/(x,) validity masks compute.

    Args:
        tile: GeoTile to crop, e.g. per-date mosaic_spatial outputs concatenated along time.

    Returns:
        New GeoTile, narrowed geobox, cropped data (still lazy if input was).

    Raises:
        ValueError: No pixel has real data across every time/band step.
    """
    if not tile.has_time:
        return tile

    valid = _mask_nodata(tile.data).notnull()
    reduce_dims = [d for d in valid.dims if d not in ("y", "x")]
    valid_2d = valid.all(dim=reduce_dims) if reduce_dims else valid
    row_mask, col_mask = valid_2d.any(dim="x"), valid_2d.any(dim="y")

    # compute(a, b) together shares valid_2d's task graph instead of computing it twice
    if is_dask_collection(row_mask.data):
        row_vals, col_vals = dask_compute(row_mask.data, col_mask.data)
    else:
        row_vals, col_vals = row_mask.values, col_mask.values

    rows, cols = np.where(row_vals)[0], np.where(col_vals)[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("align_temporal(): no pixel has real data across every time step")
    row0, row1 = int(rows.min()), int(rows.max()) + 1
    col0, col1 = int(cols.min()), int(cols.max()) + 1

    return tile.rebase(
        geobox=tile.geobox[row0:row1, col0:col1],
        data=tile.data.isel(y=slice(row0, row1), x=slice(col0, col1)),
    )


def mosaic_spatial(
    *tiles: GeoTile,
    method: MergeMethod = "first",
    target_crs: str | None = None,
    target_resolution: float | None = None,
    resampling: Resampling = Resampling.nearest,
    date_precision: DatePrecision = 'D',
) -> GeoTile:
    """Merge tiles into one, gaps filled with nodata.

    No time requirement — grouping by time is the caller's job. GeoTag
    datetime stays exact min/max regardless of date_precision. Output
    nodata is always NaN.

    Args:
        *tiles: Tiles to merge.
        method: "first" keeps tile 0's pixel wherever it has real data,
            falling through to later tiles only where tile 0 is nodata.
            "last" is the same rule with tile order reversed.
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
        ValueError: tiles is empty, method isn't "first"/"last", CRS
            mismatch with target_crs unset, resolution mismatch with
            target_resolution unset, or bands/rgb_bands/class_map/color_map
            don't match across tiles.
    """
    if not tiles:
        raise ValueError("mosaic_spatial() requires at least one tile")
    if method not in ("first", "last"):
        raise ValueError(f"mosaic_spatial(): method must be 'first' or 'last', got {method!r}")

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

    # every tile needs the same band structure and rendering/class metadata — no reconciliation for these
    has_band = "band" in tiles[0].data.dims
    for i, t in enumerate(tiles[1:], start=1):
        if ("band" in t.data.dims) != has_band:
            raise ValueError(f"mosaic_spatial(): tile {i} band dim differs from tile 0")
        if has_band and t.bands != tiles[0].bands:
            raise ValueError(f"mosaic_spatial(): tile {i} bands {t.bands} != tile 0 bands {tiles[0].bands}")
        if t.rgb_bands != tiles[0].rgb_bands:
            raise ValueError(f"mosaic_spatial(): tile {i} rgb_bands {t.rgb_bands} != tile 0 rgb_bands {tiles[0].rgb_bands}")
        if t.class_map != tiles[0].class_map:
            raise ValueError(f"mosaic_spatial(): tile {i} class_map {t.class_map} != tile 0 class_map {tiles[0].class_map}")
        if t.color_map != tiles[0].color_map:
            raise ValueError(f"mosaic_spatial(): tile {i} color_map {t.color_map} != tile 0 color_map {tiles[0].color_map}")

    # first tile's real pixels win; each later tile only fills what's still nodata so far
    ordered = tiles if method == "first" else tuple(reversed(tiles))
    merged_da = _mask_nodata(_floor_time(ordered[0].data, date_precision))
    for t in ordered[1:]:
        merged_da = merged_da.combine_first(_mask_nodata(_floor_time(t.data, date_precision)))
    # NaN needs a float dtype — combine_first only promotes it when a reindex gap forces one
    if not np.issubdtype(merged_da.dtype, np.floating):
        merged_da = merged_da.astype(np.float32)
    merged_da = merged_da.rio.write_crs(target_crs).rio.write_nodata(np.nan)

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
        polygon=polygon,
        rgb_bands=tiles[0].rgb_bands,
        class_map=tiles[0].class_map,
        color_map=tiles[0].color_map,
        **{k: v for t in tiles for k, v in t.metadata.items()},
    )
    merged_tile = GeoTile(geobox=geobox, data=merged_da, geotag=tag)
    return merged_tile.rebase(stac=[item for t in tiles for item in t.stac])


def align_temporal_stack(stack: GeoStack) -> GeoStack:
    """Apply align_temporal to every layer in stack.

    Returns:
        New GeoStack, same layer names, each cropped independently.
    """
    from .stack import GeoStack  # noqa: F811 — deferred to dodge the geostack.py <-> ops.py cycle

    return GeoStack(**{name: align_temporal(t) for name, t in stack.tiles.items()})


def mosaic_stack(*stacks: GeoStack, method: MergeMethod = "first") -> GeoStack:
    """Merge stacks into one, layer by layer, gaps filled with nodata.

    Layers don't need to match across stacks — each layer name merges
    across whichever stacks carry it.

    Args:
        *stacks: Stacks to merge.
        method: Forwarded to mosaic_spatial per layer.

    Returns:
        One GeoStack with every layer name found across inputs, no `context`
        — inputs are geometrically merged, not the same sample, so which
        stack's `temporal_coords`/etc. would even apply is undefined; `context`
        is a training-time concern (see `GeoPipeline.context`), not a spatial
        merge one. Attach it after, explicitly, via `.with_context(...)` if
        this mosaic itself is going into training.

    Raises:
        ValueError: stacks is empty, or a layer's tiles fail mosaic_spatial's own checks.

    Examples:
        >>> region = mosaic_stack(stack_a, stack_b)  # stack_b missing a layer stack_a has is fine
    """

    from .stack import GeoStack  # noqa: F811 — deferred to dodge the geostack.py <-> ops.py cycle

    if not stacks:
        raise ValueError("mosaic_stack() requires at least one stack")
    layer_names = {name for s in stacks for name in s.tiles}
    merged = {
        name: mosaic_spatial(*(s.tiles[name] for s in stacks if name in s.tiles), method=method)
        for name in layer_names
    }
    return GeoStack(**merged)


def mask_to_polygon(tile: GeoTile, drop: bool = True) -> GeoTile:
    """Mask tile's data outside its own polygon footprint, nodata-filled.

    No-op if tile.polygon is None. drop also narrows the geobox to the
    polygon's own bounds; without it the rectangular extent stays.

    Args:
        tile: GeoTile whose .polygon should become a real pixel mask.
        drop: Narrow geobox to polygon bounds too, not just mask pixels.

    Returns:
        New GeoTile, pixels outside polygon set nodata.

    Raises:
        ValueError: tile has a polygon but no CRS.
    """
    if tile.polygon is None:
        return tile
    if tile.crs is None:
        raise ValueError("mask_to_polygon() needs tile to have a real CRS")
    poly = tile.polygon.to_crs(tile.crs)
    clipped = tile.data.rio.clip([poly.geom], crs=tile.crs, drop=drop)
    return tile.rebase(geobox=clipped.odc.geobox, data=clipped)


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
