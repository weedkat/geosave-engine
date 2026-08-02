"""Free-function geometry operations on GeoTile: remap, align, mosaic, chunk_geotile.

Not GeoTile methods — keeps GeoTile's own surface to tile-specific concerns
(its own fields, its own serialization glue) instead of growing indefinitely.
Disk I/O (Zarr/GeoTIFF) and shape validation live in
`geosave_engine.geodata.utils.io`/`.geodata` — those don't need GeoTile at
all, so keeping them here would just be a needless circular-import risk.
"""
from __future__ import annotations

from typing import Literal, TypeVar, cast

import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.geom import Geometry
from rioxarray.merge import merge_datasets

from .geoanchor import GeoAnchor, GeoTag
from .geotile import GeoTile

TimeRound = Literal["D", "H", "T", "S", "L", "U", "N"]  # Pandas offset aliases
T = TypeVar("T", bound=GeoAnchor)


def remap(tile: GeoTile, mapping: dict[int, int]) -> GeoTile:
    """Return a new GeoTile with label values remapped per ``mapping``."""
    remapped = tile.data
    for src_val, dst_val in mapping.items():
        remapped = remapped.where(remapped != src_val, other=dst_val)
    return tile.rebase(data=remapped)


def align(*tiles: GeoTile) -> tuple[GeoTile, ...]:
    """Narrow each tile's geobox to their common intersection.

    Pure geometry — data is shared untouched. Tiles must share CRS,
    resolution, and pixel grid; that shared CRS must also be projected
    (metric) — `resolution` (`geobox.affine.a`) only means meters under a
    projected CRS, degrees under a geographic one (e.g. EPSG:4326), and
    nothing downstream (GSD-conditioned models, area_m2, ...) means to
    handle the degrees case.

    Raises:
        ValueError: If fewer than 2 tiles, CRS/resolution mismatch, that CRS
            isn't projected, no overlap, or misaligned grid.
    """
    if len(tiles) < 2:
        raise ValueError("align() requires at least 2 tiles")
    crss = {t.crs for t in tiles}
    if len(crss) > 1:
        raise ValueError(f"align() requires one CRS, got: {crss}")
    crs = tiles[0].geobox.crs
    if crs is None or not crs.projected:
        raise ValueError(
            f"align() requires a projected CRS (resolution must mean meters), got "
            f"{tiles[0].crs} — reproject to a projected CRS (e.g. local UTM) first"
        )
    resolutions = {round(t.resolution, 6) for t in tiles}
    if len(resolutions) > 1:
        raise ValueError(f"align() requires one resolution, got: {resolutions}")

    minx = max(t.bbox[0] for t in tiles)
    miny = max(t.bbox[1] for t in tiles)
    maxx = min(t.bbox[2] for t in tiles)
    maxy = min(t.bbox[3] for t in tiles)
    if minx >= maxx or miny >= maxy:
        raise ValueError("Tiles have no spatial overlap — cannot align")

    aligned: list[GeoTile] = []
    for t in tiles:
        res = t.resolution
        left, _, _, top = t.bbox
        col0 = (minx - left) / res
        row0 = (top - maxy) / res
        ncols = (maxx - minx) / res
        nrows = (maxy - miny) / res
        if any(abs(v - round(v)) > 1e-6 for v in (col0, row0, ncols, nrows)):
            raise ValueError("Tiles are not on a common pixel grid; reproject first")
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


def mosaic(
    tiles: list[GeoTile],
    crs: str | None = None,
    time_round_to: TimeRound = 'D',
) -> GeoTile:
    """Stitch spatially non-overlapping tiles into one larger tile.

    All tiles must share band names and time coordinates.

    Args:
        tiles: Tiles to merge.
        crs: Reproject tiles to this CRS before merging. Required if tiles differ in CRS.
        time_round_to: Pandas offset alias (e.g. "D") to floor time coords before matching.

    Raises:
        ValueError: If tiles is empty, CRS mismatch without crs=, or band/time mismatch.
    """
    if not tiles:
        raise ValueError("Cannot mosaic an empty tile list")
    if any(t.start != t.end for t in tiles):
        raise ValueError("Cannot mosaic range-datetime tiles; ingest first to resolve to single datetimes")

    tile_crss = {t.crs for t in tiles}
    if crs is None and len(tile_crss) > 1:
        raise ValueError(
            f"Cannot mosaic: tiles have different CRS: {tile_crss}. Pass crs= to reproject."
        )

    das: list[xr.DataArray] = []
    for t in tiles:
        da = t.data
        if time_round_to is not None and "time" in da.dims:
            da = da.assign_coords(time=da.time.dt.floor(time_round_to))
        if crs is not None and t.crs != crs:
            da = da.rio.reproject(crs)
        das.append(da)

    band_dims = {"band" in da.dims for da in das}
    if len(band_dims) > 1:
        raise ValueError("Cannot mosaic: some tiles have a 'band' dim and others don't")
    has_band = "band" in das[0].dims
    if has_band:
        band_sets = {tuple(str(b) for b in da.coords["band"].values) for da in das}
        if len(band_sets) > 1:
            raise ValueError(f"Cannot mosaic: tiles have different bands: {band_sets}")
    time_sets = {
        tuple(str(v) for v in da.time.values) if "time" in da.dims else ()
        for da in das
    }
    if len(time_sets) > 1:
        raise ValueError(
            "Cannot mosaic: tiles have different time steps; pass time_round_to= for tolerance"
        )

    if has_band:
        merged_ds = merge_datasets([da.to_dataset(dim="band") for da in das])
        merged = merged_ds.to_array(dim="band")
        merged = merged.transpose("time", "band", "y", "x") if "time" in merged.dims else merged.transpose("band", "y", "x")
    else:
        merged_ds = merge_datasets([da.to_dataset(name="value") for da in das])
        merged = merged_ds["value"]
        merged = merged.transpose("time", "y", "x") if "time" in merged.dims else merged.transpose("y", "x")
    geobox = GeoBox.from_bbox(
        merged.rio.bounds(),
        crs=merged.rio.crs.to_string(),
        resolution=tiles[0].resolution,
    )
    mosaic_polygon: Geometry | None = None
    tile_polys = [t.polygon for t in tiles]
    if all(p is not None for p in tile_polys):
        target_crs_str: str | None = crs or tiles[0].crs
        first_poly = tile_polys[0]
        assert first_poly is not None
        merged_poly: Geometry = first_poly
        for p in tile_polys[1:]:
            if p is not None:
                if target_crs_str is not None and str(merged_poly.crs) != target_crs_str:
                    merged_poly = merged_poly.to_crs(target_crs_str)
                merged_poly = merged_poly | p
        mosaic_polygon = merged_poly
    tag = GeoTag(
        datetime=max(t.datetime for t in tiles),
        metadata={k: v for t in tiles for k, v in t.metadata.items()},
        polygon=mosaic_polygon,
        plot_meta=tiles[0].plot_meta,
    )
    base = GeoTile(
        geobox=geobox,
        data=merged,
        geotag=tag,
    ).rebase(stac=[item for t in tiles for item in t.stac])
    return base


def chunk_geotile(full: T, tile_size_px: int) -> list[T]:
    """Chunk a GeoAnchor/GeoTile's geobox into a grid of sub-anchors.

    Args:
        full: GeoAnchor or GeoTile covering the whole area to split.
        tile_size_px: Sub-tile side length in pixels (square).

    Returns:
        One instance (same type as ``full``) per grid cell intersecting
        ``full``'s extent. ``full.polygon``, if set, is clipped to each
        cell's own bbox — not dropped — so mosaicking the same cells back
        together reconstructs the original footprint (``mosaic()`` already
        unions per-tile polygons when every tile carries one). A cell whose
        bbox falls inside ``full``'s bbox but outside the actual (non-
        rectangular) polygon clips down to an empty geometry, same as any
        other real gap.
    """
    grid = GeoboxTiles(full.geobox, (tile_size_px, tile_size_px))
    # GeoboxTiles.__getitem__ is typed to return the wider GeoBoxBase, but
    # full.geobox (and so grid, built from it) is always concretely GeoBox.
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
