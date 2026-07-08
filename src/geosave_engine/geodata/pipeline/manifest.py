from __future__ import annotations

import json
import logging
import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, TypedDict

if TYPE_CHECKING:
    from geosave_engine.geodata.core.geotile import GeoTile

log = logging.getLogger(__name__)


class LayerSpec(TypedDict, total=False):
    """Layer spec passed to ManifestWriter and layer_metadata."""

    name: str
    resolution: float
    description: str
    nodata: int | float | None
    schema: list[dict[str, Any]]


def _geo_key(anchor: GeoTile) -> str:
    """Stable spatial identity key for an anchor tile."""
    bbox = anchor.bbox
    return (
        f"{anchor.crs}"
        f"|{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
        f"|{anchor.resolution:.4f}"
        f"|{anchor.datetime.strftime('%Y%m%d')}"
    )


def layer_metadata(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a layer spec into its self-describing metadata block.

    Block is recorded in manifest and infused into every saved tile's metadata.

    Args:
        spec: LayerSpec dict with required "name" and "resolution" keys.

    Returns:
        {
            "name": str,
            "resolution": float,
            "description": str,
            "nodata": int | float | None,   # omitted if None
            "schema": list[dict],           # omitted if empty
        }.
    """
    meta: dict[str, Any] = {
        "name": spec["name"],
        "resolution": spec["resolution"],
        "description": spec.get("description", ""),
    }
    if spec.get("nodata") is not None:
        meta["nodata"] = spec["nodata"]
    if spec.get("schema"):
        meta["schema"] = list(spec["schema"])
    return meta


class ManifestWriter:
    """Ingestion tracking manifest for one layer.

    Writes <root>/<layer>/manifest.json:

        {
          "metadata": { "name", "resolution", "description", "type", "bands"|"classes" },
          "anchors": {
            "<geo_key>": {
              "stem": str,
              "source": str|null,
              "status": "pending"|"done"|"error",
              "error": str|null,
              "store": "<stem>.zarr"|null
            }
          }
        }

    Store paths are relative to the layer directory.

    Examples:
        >>> spec = {"name": "sentinel2", "resolution": 10, "bands": {"B04": {"name": "red"}}}
        >>> mw = ManifestWriter("/workspace", spec)
        >>> mw.add(anchor, stem="sentinel2_13.00_52.00_20240101_10m")
        >>> mw.mark_done(anchor, store="sentinel2_13.00_52.00_20240101_10m.zarr")
    """

    def __init__(self, root: str | Path, spec: LayerSpec) -> None:
        """
        Args:
            root: Workspace root directory.
            spec: LayerSpec with required "name" and "resolution" keys.

        Raises:
            KeyError: If spec is missing "name".
        """
        self.root = Path(root)
        self.layer = spec["name"]
        self.dir = self.root / self.layer
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "manifest.json"

        self._metadata: dict[str, Any] = {}
        self._anchors: dict[str, dict[str, Any]] = {}

        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._metadata = data.get("metadata", {})
            self._anchors = data.get("anchors", {})

        self._declare(spec)

    def _declare(self, spec: Mapping[str, Any]) -> None:
        self._metadata = layer_metadata(spec)
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"metadata": self._metadata, "anchors": self._anchors}, indent=2))
        tmp.replace(self.path)

    # --- anchor lifecycle ---

    def add(self, anchor: GeoTile, stem: str, source: str | None = None) -> None:
        """Register anchor. No-op if already registered."""
        key = _geo_key(anchor)
        if key in self._anchors:
            return
        self._anchors[key] = {
            "stem": stem,
            "source": source,
            "status": "pending",
            "error": None,
            "store": None,
        }
        self._save()

    def is_processed(self, anchor: GeoTile) -> bool:
        """Return True if anchor was already attempted (status done or error)."""
        entry = self._anchors.get(_geo_key(anchor))
        return entry is not None and entry["status"] in ("done", "error")

    def is_done(self, anchor: GeoTile) -> bool:
        """Return True if anchor is done and the zarr store exists on disk."""
        entry = self._anchors.get(_geo_key(anchor))
        if entry is None or entry["status"] != "done" or not entry["store"]:
            return False
        if not (self.dir / entry["store"]).exists():
            log.warning("Missing store %s for %s, will re-ingest", entry["store"], self.layer)
            return False
        return True

    def mark_done(self, anchor: GeoTile, store: str) -> None:
        """Record the zarr store name (relative to layer dir)."""
        entry = self._anchors[_geo_key(anchor)]
        entry["status"] = "done"
        entry["store"] = store
        self._save()

    def mark_error(self, anchor: GeoTile, message: str = "") -> None:
        """Mark anchor as failed.

        Args:
            anchor: Anchor tile to mark.
            message: Error description.

        Raises:
            KeyError: If anchor is not registered.
        """
        key = _geo_key(anchor)
        if key not in self._anchors:
            raise KeyError(f"No anchor with geo_key {key!r}")
        self._anchors[key]["status"] = "error"
        self._anchors[key]["error"] = message
        self._save()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_class_pct(geo_layer: GeoTile, class_dict: Mapping[int, str], decimal: int = 4) -> dict[str, float]:
    """Compute per-class pixel percentages from a label GeoTile.

    Args:
        geo_layer: GeoTile containing label data.
        class_dict: Maps class int values to names.
        decimal: Decimal places to round to.

    Returns:
        {
            "<class_name>": float,  # pixel fraction [0.0, 1.0]; key falls back to "class_{value}"
        }.

    Raises:
        ValueError: If geo_layer has no data.
    """
    if geo_layer.data is None:
        raise ValueError("GeoTile has no data to compute class percentages")
    values = geo_layer.data.values.flatten()
    unique, counts = np.unique(values, return_counts=True)
    total = values.size
    return {class_dict.get(u, f"class_{u}"): round(int(c) / total, decimal) for u, c in zip(unique, counts)}
