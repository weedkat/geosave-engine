"""Shared single-anchor time/location math for encoders that condition on real time/location.

Each encoder (PrithviTL, Clay, ...) formats these into its own
model_context() output shape — different models want different units
(year/day-of-year vs iso-week/hour) from the same underlying anchor.
"""
from __future__ import annotations

from datetime import datetime as dt

import numpy as np

from geosave_engine.geodata.extensions import TimeSpec
from geosave_engine.geodata.spatial import GeoAnchor


def anchor_midpoint(anchor: GeoAnchor) -> dt:
    """This anchor's (start, end) span midpoint.

    Args:
        anchor: Anchor to read the time span from.

    Returns:
        Midpoint datetime. Equals `anchor.start` when the anchor isn't a range.

    Raises:
        ValueError: The anchor declares no time span, so it has no instant
            to encode.
    """
    if anchor.start is None or anchor.end is None:
        raise ValueError(
            "this anchor is timeless, so it has no time to encode — a time-conditioned model "
            "reads its context off a dated layer, not a static one like a DEM"
        )
    return anchor.start + (anchor.end - anchor.start) / 2


def step_midpoints(anchor: GeoAnchor) -> list[dt]:
    """The instant each time step stands for, one per step.

    Args:
        anchor: Anchor to read steps from. Its header's `array` namespace
            carries the real per-step labels whenever pixels were ever
            attached, so this is exact off a window with none loaded.

    Returns:
        One midpoint per time step, in step order, each label widened to the
        bucket it names. A single midpoint from the declared span for a
        timeless anchor, or one that never carried pixels.
    """
    spec = anchor.header.array
    if spec is None or spec.times is None:
        return [anchor_midpoint(anchor)]
    buckets = (anchor.header.timespec or TimeSpec()).bounds(np.array(spec.times, dtype="datetime64[us]"))
    return [opens + (closes - opens) / 2 for opens, closes in buckets]


def year_day_of_year(anchor: GeoAnchor) -> np.ndarray:
    """(year, day-of-year) per time step — Prithvi's own time convention.

    Args:
        anchor: Anchor whose steps to derive from.

    Returns:
        `(steps, 2)` float32 — (year, day_of_year), day_of_year 0-indexed
        (Jan 1st = 0). One row for a timeless anchor.
    """
    return np.array(
        [[float(when.year), float(when.timetuple().tm_yday - 1)] for when in step_midpoints(anchor)],
        dtype="float32",
    )


def week_hour(anchor: GeoAnchor) -> np.ndarray:
    """(iso_week, hour) per time step — Clay's own time convention.

    Whole-day steps land on noon — hour carries no real intra-day signal
    unless the ingesting layer recorded a finer bucketing.

    Args:
        anchor: Anchor whose steps to derive from.

    Returns:
        `(steps, 2)` float32 — (iso_week, hour). One row for a timeless anchor.
    """
    return np.array(
        [[float(when.isocalendar().week), float(when.hour)] for when in step_midpoints(anchor)],
        dtype="float32",
    )


def lat_lon(anchor: GeoAnchor) -> tuple[float, float]:
    """(lat, lon) at this anchor's own centroid, in degrees.

    Args:
        anchor: Anchor to derive from.

    Returns:
        (lat, lon).
    """
    lon, lat = anchor.geographic_centroid
    return lat, lon
