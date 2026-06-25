from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime as dt
from pathlib import Path

from tqdm import tqdm

from .manifest import ManifestWriter, layer_metadata
from .geotile import GeoTile

log = logging.getLogger(__name__)


Result = dict[str, GeoTile]


class Pipeline(ABC):
    """Abstract base class for geospatial data pipelines.

    Subclasses define ``layer_schema`` and implement ``ingest``.
    Dedup and crash recovery are handled by AnchorTracker (anchors.json).

    Directory layout: ``root/<layer>/<tile_id>_<lon>_<lat>_<date>_<res>.tif``

    Usage:
        layer_schema = {
            "sentinel_2_l1c": {
                "resolution": 10,
                "description": "Sentinel-2 L1C imagery",
                "bands": {
                    "B02": {"name": "blue"},
                    "B03": {"name": "green"},
                    "B04": {"name": "red"},
                },
            },
            "cloud_mask": {
                "resolution": 10,
                "description": "Cloud and shadow mask",
                "classes": {
                    0: {"name": "clear", "color": "#ffffff"},
                    1: {"name": "cloud_shadow", "color": "#000000"},
                },
            },
        }
    """

    layer_schema: dict[str, Any] = {}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifests: dict[str, ManifestWriter] = {
            name: ManifestWriter(self.root, name, spec)
            for name, spec in self.layer_schema.items()
        }

    @abstractmethod
    def ingest(self, anchor: GeoTile) -> Result:
        """Fetch and process data for a spatial anchor.

        Args:
            anchor: Spatial reference defining the area and resolution.

        Returns:
            Named layers produced by this source. Each value is a single
            ``GeoTile``; time-series data uses a time-dim GeoTile.
        """
        ...
    
    def make_prefix(self, anchor: GeoTile, layer_name: str) -> str:
        """Return the prefix for the anchor-level filename.

        Override to customise naming.
        """
        return layer_name

    def make_stem(self, anchor: GeoTile, layer_name: str) -> str:  # noqa: ARG002
        """Return the anchor-level filename stem (no extension, no tile-date suffix).

        Override to customise naming. The tile-date postfix and ``.tif`` extension
        are always appended by the pipeline and must not be included here.

        Default: ``{layer}_{lon:.6f}_{lat:.6f}_{anchor_date}_{res}``
        """
        lon, lat = anchor.centroid
        res = anchor.resolution
        res_str = f"{int(res * 100)}cm" if res < 1 else f"{int(res)}m"

        prefix = self.make_prefix(anchor, layer_name)

        return f"{prefix}_{lon:.6f}_{lat:.6f}_{anchor.datetime.strftime('%Y%m%d')}_{res_str}"

    @staticmethod
    def _geo_key(anchor: GeoTile) -> str:
        """Stable spatial identity key for an anchor — layer-agnostic."""
        bbox = anchor.bbox
        return (
            f"{anchor.crs}"
            f"|{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
            f"|{anchor.resolution:.4f}"
            f"|{anchor.datetime.strftime('%Y%m%d')}"
        )

    @classmethod
    def get_layers(cls) -> list[str]:
        """Return layer names declared in ``layer_schema``."""
        return list(cls.layer_schema.keys())

    @classmethod
    def get_meta(cls, layer_name: str) -> dict:
        """Return band or class metadata dict from ``layer_schema``."""
        spec = cls.layer_schema.get(layer_name)
        if spec is None:
            return {}
        return spec.get("bands") or spec.get("classes") or {}

    @classmethod
    def get_meta_map(cls, layer_name: str, meta_name: str) -> dict:
        """Return ``{id: name}`` for a layer, or ``{}`` if not declared."""
        meta = cls.get_meta(layer_name)
        if meta is None:
            return {}
        return {k: v[meta_name] for k, v in meta.items() if meta_name in v}

    def validate_result(self, result: Result) -> Result:
        """Validate ingest() output against declared layers and schemas."""
        declared = set(self.get_layers())
        missing = declared - result.keys()
        if missing:
            raise ValueError(f"Ingest result missing declared layers: {missing}")
        extra_keys = result.keys() - declared
        if extra_keys:
            raise ValueError(f"Ingest result contains undeclared layers: {extra_keys}")

        for layer_name, tile in result.items():
            spec = self.layer_schema[layer_name]
            resolution = spec["resolution"]
            bands = spec.get("bands") or {}
            if tile.data is None:
                raise ValueError(f"Layer '{layer_name}': GeoTile has no data")
            if abs(tile.resolution - resolution) / resolution > 0.01:
                raise ValueError(
                    f"Layer '{layer_name}': resolution {tile.resolution} != spec {resolution}"
                )
            if bands:
                n_bands = tile.num_bands
                if n_bands != len(bands):
                    raise ValueError(
                        f"Layer '{layer_name}': expected {len(bands)} bands, got {n_bands}"
                    )
        return result

    def save_layer_store(self, anchor: GeoTile, layer_name: str, tile: GeoTile) -> str:
        """Write one layer's tile to a Zarr store. Returns the store name (relative to the layer dir)."""
        stem = self.make_stem(anchor, layer_name)
        store = tile.to_zarr(self.root / layer_name / f"{stem}.zarr", save_stac=True)
        return store.name

    def ingest_from_anchor(self, anchor: GeoTile, source: str) -> None:
        """Deduplicate, ingest, validate, save, and track one anchor across all layers."""
        geo_key = self._geo_key(anchor)
        stems = {name: self.make_stem(anchor, name) for name in self.manifests}
        if all(mw.is_processed(geo_key) for mw in self.manifests.values()):
            log.debug("Skipping anchor (already processed): %s", source)
            return
        for name, mw in self.manifests.items():
            mw.add(geo_key, stems[name], source=source)
        try:
            result = self.ingest(anchor)
            result = self.validate_result(result)
            for layer_name, tile in result.items():
                # infuse the layer schema (flat) so the saved tile is self-describing
                tile = tile.with_metadata(layer_metadata(layer_name, self.layer_schema[layer_name]))
                store = self.save_layer_store(anchor, layer_name, tile)
                self.manifests[layer_name].mark_done(geo_key, store)
        except Exception as e:
            log.error("Failed to ingest anchor (source=%s): %s", source, e)
            for mw in self.manifests.values():
                mw.mark_error(geo_key, str(e))

    def ingest_from_geotiff(
        self,
        src: str | Path,
        max_item: int | None = None,
    ) -> None:
        """Ingest one GeoTIFF file, or all GeoTIFFs under a directory."""
        src = Path(src)
        if src.is_dir():
            geotiffs = list(src.rglob("*.tif")) + list(src.rglob("*.tiff"))
        else:
            geotiffs = [src]

        items = geotiffs[:max_item]
        for geotiff in tqdm(items, desc=f"Ingesting {self.__class__.__name__}", unit="tile"):
            anchor = GeoTile.from_geotiff(geotiff)
            self.ingest_from_anchor(anchor, source=geotiff.name)

    def ingest_from_geojson(
        self,
        src: str | Path,
        datetime: str | dt,
        resolution: float | None = None,
    ) -> None:
        """Ingest one anchor per feature in a GeoJSON file.

        Args:
            src: Path to a GeoJSON FeatureCollection, Feature, or raw geometry.
            datetime: Acquisition datetime applied to all features.
            resolution: Pixel size in CRS units. Defaults to minimum across declared layers.
        """
        src = Path(src)
        res = resolution or self._default_resolution()

        anchors = GeoTile.from_geojson(src, resolution=res, datetime=datetime)
        for anchor in tqdm(anchors, desc=f"Ingesting {self.__class__.__name__}", unit="tile"):
            self.ingest_from_anchor(anchor, source=str(src))

    def _default_resolution(self) -> float:
        return min(spec["resolution"] for spec in self.layer_schema.values())
