"""Bulk/windowed storage beyond one store per sample.

Sibling to geodata.datasets: same underlying notion of geospatial data
(a value tied to a place, optionally a time), different access contract —
MosaicStore reads a requested window lazily out of one contiguous surface
(serving/viewing); LitDataStore packs many independent samples into a
litdata-backed store (ML input), domain-blind — it doesn't know about
GeoStack or any other type. Not "Geo"-prefixed like GeoTile/GeoAnchor/
GeoStack — these are I/O classes, not value types. See docs/concept/geotile.md.
"""
from .litdata import LitDataStore

__all__ = [
    "LitDataStore",
]
