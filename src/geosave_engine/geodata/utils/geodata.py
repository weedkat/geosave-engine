from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles

if TYPE_CHECKING:
    from geosave_engine.geodata.tile.geoanchor import GeoAnchor

T = TypeVar("T", bound="GeoAnchor")


def extract_stac_attrs(item: pystac.Item, paths: dict[str, str]) -> dict[str, Any]:
    """Extract values from a STAC item using dot-notation paths on item.to_dict().

    Args:
        item: STAC item to extract from.
        paths: Mapping of output key to dot-notation path
               (e.g. {"cloud_cover": "properties.eo:cloud_cover"}).

    Returns:
        Dict of {output_key: value}.

    Raises:
        KeyError: If a path segment is not found in the item dict.
    """
    item_dict = item.to_dict()
    result: dict[str, Any] = {}
    for output_key, dot_path in paths.items():
        node: Any = item_dict
        for seg in dot_path.split("."):
            if not isinstance(node, dict) or seg not in node:
                raise KeyError(
                    f"Path '{dot_path}' not found in STAC item '{item.id}': "
                    f"missing key '{seg}'"
                )
            node = node[seg]
        result[output_key] = node
    return result


def extract_raster_scale_offset(item: pystac.Item) -> tuple[float, float]:
    """Extract radiometric scale and offset from a STAC item's raster:bands metadata.

    Iterates over all assets and returns the first scale/offset pair found.

    Raises:
        ValueError: If no raster:bands metadata with scale/offset is found.
    """
    assets = item.to_dict().get("assets", {})
    for asset_data in assets.values():
        # 1) asset-level keys (common in some STAC catalogs)
        scale = asset_data.get("raster:scale")
        offset = asset_data.get("raster:offset")
        if scale is not None and offset is not None:
            return float(scale), float(offset)

        # 2) raster:bands list where each band may carry scale/offset
        bands = asset_data.get("raster:bands", [])
        if isinstance(bands, list):
            for band in bands:
                # band entries sometimes use 'raster:scale'/'raster:offset' or 'scale'/'offset'
                b_scale = band.get("raster:scale") if isinstance(band, dict) else None
                b_offset = band.get("raster:offset") if isinstance(band, dict) else None
                if b_scale is None:
                    b_scale = band.get("scale") if isinstance(band, dict) else None
                if b_offset is None:
                    b_offset = band.get("offset") if isinstance(band, dict) else None
                if b_scale is not None and b_offset is not None:
                    return float(b_scale), float(b_offset)

    raise ValueError(
        f"Cannot extract scale/offset from STAC item '{item.id}': "
        "no 'raster:scale'/'raster:offset' or band-level scale/offset found in any asset"
    )

def spatial_da(
    arr: np.ndarray,
    like: xr.DataArray | xr.Dataset,
) -> xr.DataArray:
    """Build a 2-D DataArray preserving spatial metadata (y, x, spatial_ref) from ``like``.

    Use this instead of bare ``xr.DataArray(arr, dims=["y","x"], coords={"y":..., "x":...})``,
    which silently drops the ``spatial_ref`` coordinate and breaks CRS detection.

    Args:
        arr: (H, W) numpy array to wrap.
        like: Source DataArray or Dataset whose y, x, and spatial_ref coords are copied.

    Returns:
        DataArray with dims ["y", "x"] and all spatial coordinates from ``like``.

    Raises:
        ValueError: If ``like`` has no y or x coordinate.
    """
    if "y" not in like.coords or "x" not in like.coords:
        raise ValueError("like must have 'y' and 'x' coordinates")

    coords: dict[str, xr.DataArray] = {"y": like.coords["y"], "x": like.coords["x"]}
    if "spatial_ref" in like.coords:
        coords["spatial_ref"] = like.coords["spatial_ref"]

    return xr.DataArray(arr, dims=["y", "x"], coords=coords)


def chunk_geotile(full: T, tile_size_px: int) -> list[T]:
    """Chunk a GeoAnchor/GeoTile's geobox into a grid of sub-anchors.

    Args:
        full: GeoAnchor or GeoTile covering the whole area to split.
        tile_size_px: Sub-tile side length in pixels (square).

    Returns:
        One instance (same type as ``full``) per grid cell intersecting
        ``full``'s extent. ``full.polygon``, if set, is dropped on each
        sub-tile — it describes the whole area, not one cell.
    """
    grid = GeoboxTiles(full.geobox, (tile_size_px, tile_size_px))
    # GeoboxTiles.__getitem__ is typed to return the wider GeoBoxBase, but
    # full.geobox (and so grid, built from it) is always concretely GeoBox.
    sub_tiles = [full.with_geobox(cast(GeoBox, grid[idx])) for idx in grid.tiles(full.geobox.extent)]
    return [dataclasses.replace(t, polygon=None) if t.polygon is not None else t for t in sub_tiles]
