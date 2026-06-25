from __future__ import annotations

import json
import logging
import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from .geotile import GeoTile

log = logging.getLogger(__name__)


class ImageSpec(TypedDict, total=False):
    """Type hint for a multiband raster layer spec. Use as a plain dict in ``layer_schema``."""
    resolution: float
    description: str
    bands: dict[str, dict[str, Any]]


class LabelSpec(TypedDict, total=False):
    """Type hint for a classification layer spec. Use as a plain dict in ``layer_schema``."""
    resolution: float
    description: str
    classes: dict[int, dict[str, Any]]


LayerSpec = ImageSpec | LabelSpec


def layer_metadata(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a layer spec into its self-describing metadata block.

    The same block is recorded in the manifest and infused (flat) into every
    saved tile's metadata, so a tile carries its own layer identity, resolution,
    and band/class schema.
    """
    if "bands" not in spec and "classes" not in spec:
        raise ValueError(f"Layer spec for {name!r} must have 'bands' or 'classes' key")
    meta: dict[str, Any] = {
        "name": name,
        "resolution": spec["resolution"],
        "description": spec.get("description", ""),
    }
    if "bands" in spec:
        meta["type"] = "image"
        meta["bands"] = {k: dict(v) for k, v in spec["bands"].items()}
    else:
        meta["type"] = "label"
        meta["classes"] = {str(k): dict(v) for k, v in spec["classes"].items()}
    return meta


class ManifestWriter:
    """One layer's self-contained manifest: ingestion tracking only.

    Created one-per-layer (a Pipeline builds one per ``layer_schema`` entry).
    Writes ``<root>/<layer>/manifest.json`` with layout:

        {
          "metadata": { "name", "resolution", "description", "type", "bands"|"classes" },
          "anchors": {
            "<geo_key>": {
              "stem": str,        // layer-specific filename stem
              "source": str|null,
              "status": "pending"|"done"|"error",
              "error": str|null,
              "store": "<stem>.zarr"|null  // zarr store dir, relative to layer dir
            }
          }
        }

    Spatial metadata is intentionally absent — read it from the Zarr store via GeoTile.
    Store paths are relative to the layer directory.
    """

    def __init__(self, root: str | Path, layer: str, spec: dict[str, Any]) -> None:
        self.root = Path(root)
        self.layer = layer
        self.dir = self.root / layer
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "manifest.json"

        self._metadata: dict[str, Any] = {}
        self._anchors: dict[str, dict[str, Any]] = {}

        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._metadata = data.get("metadata", {})
            self._anchors = data.get("anchors", {})

        self._declare(layer, spec)

    def _declare(self, name: str, spec: dict[str, Any]) -> None:
        self._metadata = layer_metadata(name, spec)
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"metadata": self._metadata, "anchors": self._anchors}, indent=2))
        tmp.replace(self.path)

    # --- anchor lifecycle ---

    def add(self, geo_key: str, stem: str, source: str | None = None) -> None:
        """Register anchor under geo_key. No-op if already registered."""
        if geo_key in self._anchors:
            return
        self._anchors[geo_key] = {
            "stem": stem,
            "source": source,
            "status": "pending",
            "error": None,
            "store": None,
        }
        self._save()

    def is_processed(self, geo_key: str) -> bool:
        """Return True if geo_key was already attempted (status done or error)."""
        entry = self._anchors.get(geo_key)
        return entry is not None and entry["status"] in ("done", "error")

    def is_done(self, geo_key: str) -> bool:
        """Return True if done and the zarr store exists on disk."""
        entry = self._anchors.get(geo_key)
        if entry is None or entry["status"] != "done" or not entry["store"]:
            return False
        if not (self.dir / entry["store"]).exists():
            log.warning("Missing store %s for %s, will re-ingest", entry["store"], self.layer)
            return False
        return True

    def mark_done(self, geo_key: str, store: str) -> None:
        """Record the zarr store name (relative to layer dir)."""
        entry = self._anchors[geo_key]
        entry["status"] = "done"
        entry["store"] = store
        self._save()

    def mark_error(self, geo_key: str, message: str = "") -> None:
        """Mark anchor as failed with optional message."""
        if geo_key not in self._anchors:
            raise KeyError(f"No anchor with geo_key {geo_key!r}")
        self._anchors[geo_key]["status"] = "error"
        self._anchors[geo_key]["error"] = message
        self._save()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_class_pct(geo_layer: GeoTile, class_dict: dict[int, str], decimal: int = 4) -> dict[str, float]:
    """Compute per-class pixel percentages from a label GeoTile.

    Args:
        geo_layer: GeoTile containing label data.
        class_dict: Maps class values to names.
        decimal: Decimal places to round to.
    """
    if geo_layer.data is None:
        raise ValueError("GeoTile has no data to compute class percentages")
    values = geo_layer.data.to_array().values.flatten()
    unique, counts = np.unique(values, return_counts=True)
    total = values.size
    return {class_dict.get(u, f"class_{u}"): round(int(c) / total, decimal) for u, c in zip(unique, counts)}
