from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


class ManifestWriter:
    """Ingestion tracking manifest for one workspace root.

    Writes <root>/manifest.json:

        {
          "metadata": {
            "pipeline": "<GeoPipeline subclass name>",
            "layers": {"<layer>": {...that layer's GeoTile.metadata...}}
          },
          "anchors": {
            "<stem>": {
              "source": str|null,
              "status": "pending"|"done"|"error",
              "error": str|null,
              "store": "<stem>"|null
            }
          }
        }

    Keyed directly by stem — the same deterministic centroid+datetime+
    resolution identity used for the anchor's own folder name, so there's
    one identity for both "have we processed this" and "what folder did we
    write."

    ``metadata`` is a snapshot from the most recent ``save_dataset`` run that
    actually ingested at least one new anchor (a fully-resumed no-op run
    leaves it untouched) — informational only, not re-verified against every
    anchor's own saved tiles. It also only reflects layers ``save_dataset``
    itself wrote; a layer added later by a separate out-of-band script (e.g.
    a label-prep step writing straight into each anchor folder) won't appear
    here even though it's really in the dataset.

    Examples:
        >>> mw = ManifestWriter("/workspace/data/train")
        >>> mw.add("13.00_52.00_20240101_10m")
        >>> mw.mark_done("13.00_52.00_20240101_10m", store="13.00_52.00_20240101_10m")
    """

    def __init__(self, root: str | Path) -> None:
        """
        Args:
            root: Workspace root directory; manifest.json written directly inside.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.json"

        self._metadata: dict[str, Any] = {}
        self._anchors: dict[str, dict[str, Any]] = {}

        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._metadata = data.get("metadata", {})
            self._anchors = data.get("anchors", {})

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"metadata": self._metadata, "anchors": self._anchors}, indent=2))
        tmp.replace(self.path)

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Attach dataset-level metadata (pipeline name, per-layer descriptions, ...).

        Overwrites any previous value — last run that ingested something wins.

        Args:
            metadata: Free-form dict, e.g. ``{"pipeline": ..., "layers": {...}}``.
        """
        self._metadata = metadata
        self._save()

    # --- anchor lifecycle ---

    def add(self, stem: str, source: str | None = None) -> None:
        """Register anchor by stem. No-op if already registered."""
        if stem in self._anchors:
            return
        self._anchors[stem] = {
            "source": source,
            "status": "pending",
            "error": None,
            "store": None,
        }
        self._save()

    def is_processed(self, stem: str) -> bool:
        """Return True if anchor was already attempted (status done or error)."""
        entry = self._anchors.get(stem)
        return entry is not None and entry["status"] in ("done", "error")

    def mark_done(self, stem: str, store: str) -> None:
        """Record the anchor's folder name (relative to root)."""
        entry = self._anchors[stem]
        entry["status"] = "done"
        entry["store"] = store
        self._save()

    def mark_error(self, stem: str, message: str = "") -> None:
        """Mark anchor as failed.

        Args:
            stem: Anchor's stem, as passed to add().
            message: Error description.

        Raises:
            KeyError: If stem is not registered.
        """
        if stem not in self._anchors:
            raise KeyError(f"No anchor with stem {stem!r}")
        self._anchors[stem]["status"] = "error"
        self._anchors[stem]["error"] = message
        self._save()
