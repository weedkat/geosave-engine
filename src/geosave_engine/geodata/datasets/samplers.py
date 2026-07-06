from __future__ import annotations

import abc
import logging
from datetime import timedelta
from typing import Any, Iterable, Mapping

import geopandas as gpd
import torch
from shapely.geometry import box

from geosave_engine.geodata.core import GeoTile, align

log = logging.getLogger(__name__)

LayerName = str
AnchorGroup = dict[LayerName, GeoTile]  # one co-located tile per layer
WGS84 = "EPSG:4326"


def stack_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack list of sample dicts into one batched dict for DataLoader.

    Tensor values are stacked on dim 0; dict values collated recursively; non-tensor values gathered as list.

    Args:
        samples: Iterable of sample dicts from GeoDataset.__getitem__.

    Returns:
        {
            "<key>": torch.Tensor,  # stacked if value was Tensor
            "<key>": dict,          # recursively collated if value was dict
            "<key>": list,          # gathered as list otherwise
        }.
    """
    sample_list = list(samples)
    sample_keys = set(sample_list[0].keys())
    out: dict[str, Any] = {}
    for key in sample_keys:
        values = [s[key] for s in sample_list]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        elif isinstance(values[0], dict):
            out[key] = stack_samples(values)
        else:
            out[key] = values
    return out


def patch_tile(tile: GeoTile, size: int, stride: int) -> list[GeoTile]:
    """Slide window over tile grid. No pixels read — geometry only.

    Args:
        tile: Source tile.
        size: Patch height and width in pixels.
        stride: Step between patches.

    Returns:
        List of lazy patch tiles sharing parent data.
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
    """Spatially join per-layer tile frames into co-located sample rows.

    Drops edge-touch overlaps, mixed-CRS rows, and datetime mismatches.

    Args:
        catalog: Layer name → GeoDataFrame with ``[geometry, tile]`` columns (WGS84).
        datetime_tol: Max datetime gap between layers. ``None`` skips check.

    Returns:
        GeoDataFrame with ``geometry`` (WGS84 intersection) + one GeoTile column per layer.
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
            dts = [t.ref_datetime for t in tiles]
            if any(abs(dts[0] - dt) > datetime_tol for dt in dts[1:]):
                continue

        aligned = align(*tiles) if len(tiles) > 1 else tiles
        rows.append({"geometry": row.geometry, **dict(zip(layers, aligned))})

    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=WGS84)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)


class GeoTileSampler(abc.ABC):
    """Build sample index from per-layer catalog.

    Subclasses define how catalog rows become sample rows.
    """

    @abc.abstractmethod
    def build_index(self, catalog: dict[LayerName, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """Join the catalog into the sample-row GeoDataFrame."""


class PreChippedSampler(GeoTileSampler):
    """One sample per co-located group — each chip used whole."""

    def __init__(self, datetime_tol: timedelta | None = None) -> None:
        """
        Args:
            datetime_tol: Max datetime gap between layers. ``None`` skips check.
        """
        self.datetime_tol = datetime_tol

    def build_index(self, catalog: dict[LayerName, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        return colocate(catalog, self.datetime_tol)


class GridSampler(GeoTileSampler):
    """Sliding-window sampler. Explodes each co-located group into patch rows."""

    def __init__(
        self,
        patch_size: int,
        stride: int | None = None,
        datetime_tol: timedelta | None = None,
    ) -> None:
        """
        Args:
            patch_size: Patch height and width in pixels.
            stride: Step between patches. Defaults to ``patch_size`` (no overlap).
            datetime_tol: Max datetime gap between layers. ``None`` skips check.
        """
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
