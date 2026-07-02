from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .manifest import ManifestWriter, LayerSpec, layer_metadata
from .geotile import GeoTile
from .specs import AnyIngestSource

log = logging.getLogger(__name__)


class Pipeline(ABC):
    """Abstract base class for single-layer geospatial data pipelines.

    Each subclass writes exactly one layer. Chain pipelines by passing
    a layer's zarr dir to the next pipeline's ``ingest_from_zarr``.

    Directory layout: ``<root>/<layer_name>/<stem>.zarr``

    Subclass must declare class attributes:

        layer_name  — layer directory name and tile identifier
        resolution  — pixel size in CRS units
        description — human-readable layer description (optional)
        nodata      — nodata/ignore value (optional)
        schema      — list of ``{"id": ..., "name": ..., ...}`` dicts (optional)

    Examples:
        class Sentinel2Pipeline(Pipeline):
            layer_name = "sentinel_2_l1c"
            resolution = 10
            description = "Sentinel-2 L1C imagery"
            schema = [
                {"id": "B02", "name": "blue"},
                {"id": "B04", "name": "red"},
            ]

            def ingest(self, anchor: GeoTile) -> GeoTile:
                ...
    """

    layer_name: str
    resolution: float
    description: str = ""
    nodata: int | float | None = None
    schema: list[dict[str, Any]] = []

    def __init__(self, root: str | Path) -> None:
        """
        Args:
            root: Workspace root directory; layer subdir is created inside.
        """
        self.root = Path(root)
        spec: LayerSpec = {
            "name": self.layer_name,
            "resolution": self.resolution,
            "description": self.description,
            "nodata": self.nodata,
            "schema": self.schema,
        }
        self.manifest = ManifestWriter(self.root, spec)

    @abstractmethod
    def ingest(self, anchor: GeoTile) -> GeoTile:
        """Fetch or derive data for one anchor tile.

        Args:
            anchor: Spatial reference defining area and resolution.
                    May carry loaded data when chained via ingest_from_zarr.

        Returns:
            GeoTile with data for this pipeline's layer.
        """
        ...

    def make_prefix(self, anchor: GeoTile) -> str:  # noqa: ARG002
        """Return filename prefix. Defaults to layer_name. Override to customise."""
        return self.layer_name

    def make_stem(self, anchor: GeoTile) -> str:
        """Return anchor-level filename stem (no extension).

        Args:
            anchor: Spatial reference tile.

        Returns:
            Stem string, e.g. ``sentinel_2_l1c_13.000000_52.000000_20240101_10m``.
        """
        lon, lat = anchor.centroid
        res = anchor.resolution
        res_str = f"{int(res * 100)}cm" if res < 1 else f"{int(res)}m"
        prefix = self.make_prefix(anchor)
        return f"{prefix}_{lon:.6f}_{lat:.6f}_{anchor.datetime.strftime('%Y%m%d')}_{res_str}"

    def get_meta_map(self, field: str) -> dict:
        """Return ``{id: field_value}`` for all schema entries that have ``field``.

        Args:
            field: Key to extract from each schema entry (e.g. ``"name"``).

        Returns:
            ``{"B02": "blue", "B04": "red"}`` or ``{0: "#ffffff", 1: "#000000"}``.

        Examples:
            >>> pipeline.get_meta_map("name")
            {'B02': 'blue', 'B04': 'red'}
        """
        return {item["id"]: item[field] for item in self.schema if field in item}

    def validate(self, tile: GeoTile) -> GeoTile:
        """Validate ingest() output against declared layer spec.

        Args:
            tile: GeoTile returned by ingest().

        Returns:
            Same tile if valid.

        Raises:
            ValueError: If tile has no data, wrong resolution, or wrong band count.
        """
        if tile.data is None:
            raise ValueError(f"Layer '{self.layer_name}': GeoTile has no data")
        if abs(tile.resolution - self.resolution) / self.resolution > 0.01:
            raise ValueError(
                f"Layer '{self.layer_name}': resolution {tile.resolution} != spec {self.resolution}"
            )
        # Band count check applies only to image layers (schema ids are str).
        # Class schemas (int ids) describe pixel values, not band count.
        if self.schema and isinstance(self.schema[0].get("id"), str):
            if tile.num_bands != len(self.schema):
                raise ValueError(
                    f"Layer '{self.layer_name}': expected {len(self.schema)} bands, got {tile.num_bands}"
                )
        return tile

    def save_layer(self, anchor: GeoTile, tile: GeoTile) -> str:
        """Write tile to zarr store under ``<root>/<layer_name>/``.

        Args:
            anchor: Reference tile used to compute the stem filename.
            tile: GeoTile to write.

        Returns:
            Store directory name relative to the layer dir (e.g. ``stem.zarr``).
        """
        stem = self.make_stem(anchor)
        store = tile.to_zarr(self.root / self.layer_name / f"{stem}.zarr", save_stac=True)
        return store.name

    def _spec_dict(self) -> dict[str, Any]:
        return {
            "name": self.layer_name,
            "resolution": self.resolution,
            "description": self.description,
            "nodata": self.nodata,
            "schema": self.schema,
        }

    def ingest_from_anchor(self, anchor: GeoTile, source: str) -> None:
        """Deduplicate, ingest, validate, save, and track one anchor.

        Args:
            anchor: Spatial reference tile defining area and resolution.
            source: Human-readable provenance string recorded in the manifest.
        """
        if self.manifest.is_processed(anchor):
            log.debug("Skipping anchor (already processed): %s", source)
            return
        stem = self.make_stem(anchor)
        self.manifest.add(anchor, stem, source=source)
        try:
            tile = self.ingest(anchor)
            tile = self.validate(tile)
            tile = tile.with_metadata(layer_metadata(self._spec_dict()), replace=True)
            store = self.save_layer(anchor, tile)
            self.manifest.mark_done(anchor, store)
        except Exception as e:
            log.error("Failed to ingest anchor (source=%s): %s", source, e)
            self.manifest.mark_error(anchor, str(e))

    def _expand_anchor(self, anchor: GeoTile) -> list[GeoTile]:
        """Expand a range-datetime anchor into real single-datetime anchors.

        Base implementation passes through single-datetime anchors unchanged.
        Subclasses that have a STAC client (e.g. Sentinel2Pipeline) override this
        to query available scenes within the date range.

        Raises:
            TypeError: If anchor carries a date range and the subclass has no STAC expansion.
        """
        if isinstance(anchor.datetime, tuple):
            raise TypeError(
                f"{type(self).__name__} cannot expand date-range anchors; "
                "override _expand_anchor in a STAC-capable pipeline subclass"
            )
        return [anchor]

    def ingest_from(self, src: AnyIngestSource, max_item: int | None = None) -> None:
        """Ingest anchors produced by a typed source spec.

        Chain pipelines by passing a ``ZarrSource`` pointing at the upstream
        layer directory. For new data use ``GeoJSONSource``, ``CoordinateSource``,
        ``PolygonSource``, or ``GeotiffSource``. Range-datetime anchors are expanded
        into real single-datetime anchors via ``_expand_anchor`` before ingestion.

        Args:
            src: Typed source spec; call ``src.to_anchors(resolution)`` to get anchors.
            max_item: Cap on anchors to process; None means all.
        """
        raw_anchors = src.to_anchors(limit=None)
        expanded: list[GeoTile] = []
        for anchor in raw_anchors:
            expanded.extend(self._expand_anchor(anchor))
        if max_item is not None:
            expanded = expanded[:max_item]
        for anchor in tqdm(expanded, desc=f"Ingesting {self.__class__.__name__}", unit="tile"):
            self.ingest_from_anchor(anchor, source=repr(src))
        
    @classmethod
    def band_map(cls) -> dict[str, str]:
        """Return ``{band_id: band_name}`` from schema."""
        return {str(item["id"]): str(item["name"]) for item in cls.schema if "name" in item}

    @classmethod
    def class_map(cls) -> dict[int, str]:
        """Return ``{id: name}`` from schema."""
        return {int(item["id"]): str(item["name"]) for item in cls.schema if "name" in item}

    @classmethod
    def color_map(cls) -> dict[int, str]:
        """Return ``{id: color}`` from schema."""
        return {int(item["id"]): str(item["color"]) for item in cls.schema if "color" in item}
