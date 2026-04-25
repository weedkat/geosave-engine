"""GeoPackage manifest I/O — layer-agnostic.

A geosave manifest is a single GeoPackage file containing many layers:
- Geometry layers (one per dataset) hold per-sample records.
- Attribute layers (no geometry) hold metadata such as class tables or band info.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from pyogrio import list_layers

from geosave_engine.utils.geodata.labels import build_remap_from_class_meta

MANIFEST_CRS = "EPSG:4326"
DRIVER = "GPKG"

CLASS_META_REQUIRED_COLS = ("class_id", "class_name", "source_class_id", "ignore")
BAND_META_REQUIRED_COLS = ("band_index", "band_name")


@dataclass(frozen=True)
class TrainingMeta:
    """Class-table-derived training parameters.

    Attributes:
        num_classes: count of non-ignored classes (the model's output channels).
        ignore_index: the training class id used to mark ignored pixels.
        class_names: ordered list of non-ignored class names by training class_id.
        class_id_map: source-pixel-value → training class_id remap (covers ignored
            classes too, mapping them to ``ignore_index``).
    """

    num_classes: int
    ignore_index: int
    class_names: list[str]
    class_id_map: dict[int, int]


def list_manifest_layers(path: Path) -> list[str]:
    """Return all layer names in the GeoPackage; empty list if file absent."""
    if not Path(path).exists():
        return []
    return [name for name, _ in list_layers(path)]


def append_geo_layer(
    record: dict,
    path: Path,
    layer: str,
    native_crs: str | int,
) -> None:
    """Append one record to a geometry layer; transform geometry to WGS84."""
    row = gpd.GeoDataFrame([record], crs=native_crs).to_crs(MANIFEST_CRS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if Path(path).exists() else "w"
    row.to_file(path, driver=DRIVER, mode=mode, engine="pyogrio", layer=layer)


def load_geo_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    """Load a geometry layer."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    return gpd.read_file(path, layer=layer, driver=DRIVER, engine="pyogrio")


def write_attribute_layer(
    rows: list[dict],
    path: Path,
    layer: str,
    *,
    overwrite: bool = False,
) -> None:
    """Write rows to a non-geometry attribute layer.

    Idempotent by default: skips silently if the layer already exists. Pass
    ``overwrite=True`` to replace it.
    """
    existing = list_manifest_layers(path)
    if layer in existing and not overwrite:
        return

    gdf = gpd.GeoDataFrame(rows, geometry=[None] * len(rows))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if Path(path).exists() else "w"
    gdf.to_file(path, driver=DRIVER, mode=mode, engine="pyogrio", layer=layer)


def load_attribute_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    """Load a non-geometry attribute layer."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    return gpd.read_file(path, layer=layer, engine="pyogrio")


def load_class_meta(path: Path, layer: str) -> gpd.GeoDataFrame:
    """Load a class-table attribute layer; validate required columns."""
    df = load_attribute_layer(path, layer)
    missing = [c for c in CLASS_META_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"class meta layer '{layer}' missing required columns: {missing}"
        )
    return df


def load_band_meta(path: Path, layer: str) -> gpd.GeoDataFrame:
    """Load a band-index attribute layer; validate required columns."""
    df = load_attribute_layer(path, layer)
    missing = [c for c in BAND_META_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"band meta layer '{layer}' missing required columns: {missing}"
        )
    return df.sort_values("band_index").reset_index(drop=True)


def derive_training_meta(class_meta: gpd.GeoDataFrame) -> TrainingMeta:
    """Derive training-time class parameters from a class-table layer."""
    ignore_mask = class_meta["ignore"].astype(bool)
    ignored = class_meta[ignore_mask]
    if ignored.empty:
        raise ValueError("class meta has no ignore row; expected exactly one")
    if len(ignored) > 1:
        raise ValueError(
            f"class meta has {len(ignored)} ignore rows; expected exactly one"
        )

    kept = class_meta[~ignore_mask].sort_values("class_id")
    class_names = kept["class_name"].astype(str).tolist()
    class_id_map = build_remap_from_class_meta(class_meta.to_dict("records"))

    return TrainingMeta(
        num_classes=int(len(kept)),
        ignore_index=int(ignored["class_id"].iloc[0]),
        class_names=class_names,
        class_id_map=class_id_map,
    )
