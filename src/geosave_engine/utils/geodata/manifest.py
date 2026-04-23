"""GeoDataFrame manifest for tracking ingested samples."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyogrio import list_layers

MANIFEST_CRS = "EPSG:4326"
DRIVER = "GPKG"


def load_manifest(path: Path) -> gpd.GeoDataFrame:
    """Load an existing GeoPackage manifest."""
    if path.exists():
        return gpd.read_file(path, layer="manifest", driver=DRIVER, engine="pyogrio")
    raise FileNotFoundError(f"Manifest file not found: {path}")


def append_to_manifest(
    record: dict,
    path: Path,
    native_crs: str | int,
) -> None:
    """Append one record, transform geometry column to WGS84, and save."""
    row = gpd.GeoDataFrame([record], crs=native_crs).to_crs(MANIFEST_CRS)
    mode = "a" if path.exists() else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_file(path, driver=DRIVER, mode=mode, engine="pyogrio", layer="manifest")


def write_class_meta(classes: list[dict], path: Path) -> None:
    """Write class metadata to the 'classes' layer of a GeoPackage.

    Each dict must have: class_id, class_name, color_r, color_g, color_b.
    Skips silently if the 'classes' layer already exists.
    """
    if path.exists() and "classes" in list_layers(path)[:, 0]:
        return

    gdf = gpd.GeoDataFrame(classes, geometry=[None] * len(classes))
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    gdf.to_file(path, driver=DRIVER, mode=mode, engine="pyogrio", layer="classes")


def load_class_meta(path: Path) -> gpd.GeoDataFrame:
    """Load the 'classes' layer from a GeoPackage manifest."""
    return gpd.read_file(path, layer="classes", engine="pyogrio")
