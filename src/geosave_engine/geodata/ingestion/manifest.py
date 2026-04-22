"""GeoDataFrame manifest for tracking ingested samples."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

MANIFEST_CRS = "EPSG:4326"
MANIFEST_COLUMNS = [
    "dw_id", "labeler", "item_id", "sensing_datetime", "sun_azimuth",
    "masked_pct", "s2c_pct", "cdi_pct", "b10_pct", "shadow_pct",
    "input_path", "mask_path", "tci_path", "split",
]


def load_or_init_manifest(path: Path, columns: list[str] = MANIFEST_COLUMNS) -> gpd.GeoDataFrame:
    """Load an existing GeoJSON manifest or return an empty one with the correct schema."""
    if path.exists():
        return gpd.read_file(path)

    return gpd.GeoDataFrame(
        columns=columns,
        geometry=gpd.GeoSeries(dtype="geometry"),
        crs=MANIFEST_CRS,
    )


def append_to_manifest(
    gdf: gpd.GeoDataFrame,
    record: dict,
    path: Path,
) -> gpd.GeoDataFrame:
    """Append one record, save to disk, and return the updated GeoDataFrame.

    ``record`` must contain a ``'geometry'`` key holding a shapely geometry
    in WGS-84 (EPSG:4326).
    """
    new_row = gpd.GeoDataFrame([record], geometry="geometry", crs=MANIFEST_CRS)
    gdf = gpd.GeoDataFrame(
        pd.concat([gdf, new_row], ignore_index=True),
        geometry="geometry",
        crs=MANIFEST_CRS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    return gdf
