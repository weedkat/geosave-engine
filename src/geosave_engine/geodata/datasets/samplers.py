from __future__ import annotations

import abc
import logging
from datetime import timedelta

import geopandas as gpd
from shapely.geometry import box

from geosave_engine.geodata.core import GeoTile, align

log = logging.getLogger(__name__)

LayerName = str
AnchorGroup = dict[LayerName, GeoTile]  # one co-located tile per layer
WGS84 = "EPSG:4326"


def patch_tile(tile: GeoTile, size: int, stride: int) -> list[GeoTile]:
    """Slide a window over a tile's grid → lazy patch tiles.

    Each patch shares the parent's lazy ``data`` with only a narrowed sub-geobox,
    so ``to_tensor`` later reads just the patch's extent. Pure geometry: no pixels
    read or copied here.

    Args:
        tile: Source tile (header-only or data-loaded).
        size: Patch height and width in pixels.
        stride: Step between patches.
    """
    h, w = tile.height, tile.width
    patches: list[GeoTile] = []
    for y0 in range(0, max(1, h - size + 1), stride):
        for x0 in range(0, max(1, w - size + 1), stride):
            patches.append(tile.with_geobox(tile.geobox[y0 : y0 + size, x0 : x0 + size]))
    return patches


def colocate(
    catalog: dict[LayerName, gpd.GeoDataFrame],
    datetime_tol: timedelta | None = None,
) -> gpd.GeoDataFrame:
    """Spatially join per-layer tile frames into a co-located sample GeoDataFrame.

    Each input frame has columns ``[geometry, tile]`` (WGS84 footprints). Layers
    are joined via ``overlay(intersection)`` so the result geometry is the actual
    overlap polygon (not the left tile's bbox). ``keep_geom_type=True`` drops
    edge-touching neighbours whose intersection is a line or point. Rows with mixed
    CRS across layers are skipped. If ``datetime_tol`` is set, rows where any two
    tiles' anchor datetimes differ beyond the tolerance are also dropped. Surviving
    rows have their tiles aligned to the common pixel window.

    Returns a GeoDataFrame with ``geometry`` (WGS84 intersection) plus one aligned
    ``GeoTile`` column per layer, one row per sample.
    """
    layers = list(catalog)
    if not layers:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=WGS84)

    joined = catalog[layers[0]].rename(columns={"tile": layers[0]})
    for layer in layers[1:]:
        right = catalog[layer].rename(columns={"tile": layer})
        joined = gpd.overlay(joined, right, how="intersection", keep_geom_type=True)

    rows = []
    for _, row in joined.iterrows():
        tiles = [row[layer] for layer in layers]

        if len({t.crs for t in tiles}) > 1:
            log.warning("Skipping row with mixed CRS across layers: %s", [t.crs for t in tiles])
            continue

        # WGS84 bboxes of projected tiles (e.g. UTM) are oversized, so overlay may
        # match adjacent tiles that don't actually overlap in native CRS. Check here.
        native_overlap = box(*tiles[0].bbox)
        for t in tiles[1:]:
            native_overlap = native_overlap.intersection(box(*t.bbox))
        if native_overlap.area <= 0:
            continue

        if datetime_tol is not None:
            dts = [t.datetime for t in tiles]
            if any(abs(dts[0] - dt) > datetime_tol for dt in dts[1:]):
                continue

        aligned = align(*tiles) if len(tiles) > 1 else tiles
        rows.append({"geometry": row.geometry, **dict(zip(layers, aligned))})

    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=WGS84)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)


class GeoTileSampler(abc.ABC):
    """Builds the sample index from a per-layer catalog.

    ``build_index`` joins the catalog ``{layer: frame}`` into a single
    GeoDataFrame whose rows are samples (``geometry`` + one ``GeoTile`` column per
    layer, tiles aligned). PreChipped emits one row per co-located group; Grid
    explodes each group into windowed patch rows.
    """

    @abc.abstractmethod
    def build_index(self, catalog: dict[LayerName, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """Join the catalog into the sample-row GeoDataFrame."""


class PreChippedSampler(GeoTileSampler):
    """One sample per co-located group — each chip is used whole."""

    def __init__(self, datetime_tol: timedelta | None = None) -> None:
        self.datetime_tol = datetime_tol

    def build_index(self, catalog: dict[LayerName, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        return colocate(catalog, self.datetime_tol)


class GridSampler(GeoTileSampler):
    """Sliding-window sampler: explodes each co-located group into patch rows.

    Groups are aligned by :func:`colocate`, so every layer slices on the same grid
    and each patch row holds the co-located window across layers.

    Args:
        patch_size: Patch height and width in pixels.
        stride: Step between patches. Defaults to ``patch_size`` (no overlap).
        datetime_tol: Passed to :func:`colocate`; ``None`` skips datetime check.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int | None = None,
        datetime_tol: timedelta | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.datetime_tol = datetime_tol

    def build_index(self, catalog: dict[LayerName, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        sample_gdf = colocate(catalog, self.datetime_tol)
        layers = list(catalog)
        rows = []
        for _, row in sample_gdf.iterrows():
            per_layer = {
                layer: patch_tile(row[layer], self.patch_size, self.stride)
                for layer in layers
            }
            counts = {layer: len(p) for layer, p in per_layer.items()}
            if len(set(counts.values())) > 1:
                raise ValueError(
                    f"Layers produced different patch counts {counts}; "
                    "group geoboxes are not aligned"
                )
            n = next(iter(counts.values()))
            for i in range(n):
                patch_group = {layer: per_layer[layer][i] for layer in layers}
                rows.append({
                    "geometry": box(*next(iter(patch_group.values())).wgs84_bbox),
                    **patch_group,
                })
        if not rows:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=WGS84)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)
