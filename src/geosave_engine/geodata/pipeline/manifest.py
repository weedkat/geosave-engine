"""GeoParquet ledger of what a pipeline has ingested under one root."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from os import PathLike
    from typing import Self

    from geosave_engine.geodata.core import GeoAnchor
    from geosave_engine.geodata.utils.geo.geolocator import Place

_FILENAME = "manifest.parquet"
_COLUMNS = (
    "anchor_id",
    "stem",
    "status",
    "output",
    "crs",
    "bounds",
    "shape",
    "resolution",
    "timespan_start",
    "timespan_end",
    "bytes",
    "address",
    "state_province",
    "country",
    "country_code",
    "error",
    "ingested_at",
)


def _anchor_id(anchor: GeoAnchor) -> str:
    """Return the manifest key for `anchor`.

    Hashes the stem, CRS, and timespan, so anchors that differ only by CRS —
    which the stem alone does not capture — get distinct keys.

    Args:
        anchor: Anchor to identify.

    Returns:
        Sixteen hex characters keying this anchor in the manifest.
    """
    key = f"{anchor.stem}|{anchor.geobox.crs}|{anchor.timespan}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _empty_frame() -> gpd.GeoDataFrame:
    """Return the manifest's zero-row frame with its full schema.

    Returns:
        Empty GeoDataFrame carrying every manifest column and an EPSG:4326
        geometry column.
    """
    columns = {name: pd.Series(dtype="object") for name in _COLUMNS}
    return gpd.GeoDataFrame(
        {**columns, "geometry": gpd.GeoSeries([], crs="EPSG:4326")},
        geometry="geometry",
        crs="EPSG:4326",
    )


def _merged(frame: gpd.GeoDataFrame, buffer: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    """Return `frame` with `buffer` appended.

    Args:
        frame: Rows already flushed.
        buffer: Rows recorded since the last flush.

    Returns:
        A GeoDataFrame — the concatenation is rewrapped so the geometry column
        and CRS survive — or `frame` unchanged when `buffer` is empty.
    """
    if not buffer:
        return frame
    added = gpd.GeoDataFrame(buffer, geometry="geometry", crs="EPSG:4326")
    if frame.empty:
        return added
    return gpd.GeoDataFrame(
        pd.concat([frame, added], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )


class IngestManifest:
    """Resumable GeoParquet ledger of one ingest run.

    One row per attempted anchor: key, status, the store it wrote, the anchor
    grid, an EPSG:4326 footprint, and an optional geocoded place.
    `record_ok`/`record_failure` buffer rows; `flush` writes the file atomically.

    Args:
        root: Directory the ingest writes under; the manifest lives at
            ``<root>/manifest.parquet``.
        frame: Rows already on disk, or an empty frame for a new run.
    """

    def __init__(self, root: Path, frame: gpd.GeoDataFrame) -> None:
        """Bind a loaded frame to its root; construct through `open`."""
        self._root = Path(root)
        self._frame = frame
        self._buffer: list[dict[str, Any]] = []
        self._keys: set[str] = (
            set() if frame.empty else set(frame["anchor_id"].tolist())
        )

    @classmethod
    def open(cls, root: str | PathLike[str]) -> Self:
        """Load the manifest under `root`, or start an empty one.

        Args:
            root: Directory the ingest writes under.

        Returns:
            Manifest holding every row previously flushed under `root`.
        """
        base = Path(root)
        path = base / _FILENAME
        frame = gpd.read_parquet(path) if path.exists() else _empty_frame()
        return cls(base, frame)

    def has(self, anchor: GeoAnchor) -> bool:
        """Report whether `anchor` was already recorded, in any status.

        Args:
            anchor: Anchor to check.

        Returns:
            True when a prior attempt recorded this anchor, so a resume skips
            it. A failed anchor counts as recorded; drop its row to retry.
        """
        return _anchor_id(anchor) in self._keys

    def record_ok(
        self,
        anchor: GeoAnchor,
        *,
        output: str | PathLike[str],
        bytes_written: int,
        place: Place | None = None,
    ) -> None:
        """Buffer a success row for `anchor`.

        Grid columns and the EPSG:4326 footprint are derived from the anchor.

        Args:
            anchor: Anchor that was ingested.
            output: Store path, relative to `root`.
            bytes_written: Total size of the written store.
            place: Reverse-geocoded location, when the driver resolved one.
        """
        self._append(
            anchor,
            place,
            status="ok",
            output=str(output),
            size=int(bytes_written),
            error=None,
        )

    def record_failure(
        self,
        anchor: GeoAnchor,
        *,
        error: str,
        place: Place | None = None,
    ) -> None:
        """Buffer a failure row for `anchor`.

        Args:
            anchor: Anchor whose ingest failed.
            error: Failure detail.
            place: Reverse-geocoded location, when the driver resolved one.
        """
        self._append(anchor, place, status="failed", output=None, size=0, error=error)

    def flush(self) -> Path | None:
        """Write buffered rows to ``<root>/manifest.parquet`` atomically.

        A temporary file in the same directory is renamed over the target, so
        a crash never leaves a partial manifest. Does nothing when the buffer
        is empty.

        Returns:
            The manifest path when rows were written, else None.
        """
        if not self._buffer:
            return None
        self._frame = _merged(self._frame, self._buffer)
        self._buffer.clear()

        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / _FILENAME
        staged = path.with_name(f"{_FILENAME}.tmp")
        self._frame.to_parquet(staged)
        os.replace(staged, path)
        return path

    def frame(self) -> gpd.GeoDataFrame:
        """Return every row, flushed and buffered.

        Returns:
            GeoDataFrame with footprints in EPSG:4326, for a dataset builder to
            enumerate and split without reading pixels.
        """
        return _merged(self._frame, self._buffer)

    def overlaps(self, anchor: GeoAnchor) -> gpd.GeoDataFrame:
        """Return recorded rows whose footprint intersects `anchor`.

        For reviewing redundant coverage; not a resume gate, since overlapping
        anchors are legitimate.

        Args:
            anchor: Anchor to test against every recorded footprint.

        Returns:
            The intersecting rows, in record order.
        """
        rows = self.frame()
        if rows.empty:
            return rows
        footprint = anchor.geobox.extent.to_crs("EPSG:4326").geom
        return rows.iloc[sorted(rows.sindex.query(footprint, predicate="intersects"))]

    @property
    def pending(self) -> int:
        """Return the number of buffered rows not yet flushed."""
        return len(self._buffer)

    def _append(
        self,
        anchor: GeoAnchor,
        place: Place | None,
        *,
        status: str,
        output: str | None,
        size: int,
        error: str | None,
    ) -> None:
        """Build one row from the anchor and buffer it, keying it for `has`."""
        geobox = anchor.geobox
        span = anchor.timespan
        located = place.to_dict() if place is not None else {}
        anchor_id = _anchor_id(anchor)
        self._buffer.append(
            {
                "anchor_id": anchor_id,
                "stem": anchor.stem,
                "status": status,
                "output": output,
                "crs": str(geobox.crs),
                "bounds": [float(edge) for edge in geobox.boundingbox],
                "shape": list(geobox.shape.yx),
                "resolution": list(anchor.resolution.map(abs).xy),
                "timespan_start": None if span is None else span[0].isoformat(),
                "timespan_end": None if span is None else span[1].isoformat(),
                "bytes": size,
                "address": located.get("address"),
                "state_province": located.get("state/province"),
                "country": located.get("country"),
                "country_code": located.get("country_code"),
                "error": error,
                "ingested_at": datetime.now(UTC).isoformat(),
                "geometry": geobox.extent.to_crs("EPSG:4326").geom,
            }
        )
        self._keys.add(anchor_id)
