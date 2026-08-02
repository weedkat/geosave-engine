"""Reverse geocoding utilities backed by Nominatim (OpenStreetMap).

Results are cached per (lat, lon) rounded to 2 decimal places (~1 km grid).
All returned names use English/Latin script via ``language="en"``.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, fields

from .crs import validate_coordinate

_ADMIN_SUFFIXES = re.compile(
    r"\s+(Province|State|Region|District|Prefecture|Department|Governorate|Oblast|Krai|County|Municipality)$",
    re.IGNORECASE,
)


def _strip_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    return _ADMIN_SUFFIXES.sub("", value).strip() or None


@dataclass
class Place:
    """Raw Nominatim address components stored per OSM key.

    Use ``to_dict()`` for manifest columns or ``to_address()`` for a human-readable string.
    """

    # City-level populated places
    city: str | None = None
    town: str | None = None
    village: str | None = None
    hamlet: str | None = None
    isolated_dwelling: str | None = None
    # Sub-city
    suburb: str | None = None
    borough: str | None = None
    quarter: str | None = None
    neighbourhood: str | None = None
    city_district: str | None = None
    # County / district
    county: str | None = None
    municipality: str | None = None
    district: str | None = None
    state_district: str | None = None
    # State / province
    state: str | None = None
    province: str | None = None
    region: str | None = None
    # Country
    country: str | None = None
    country_code: str | None = None
    iso_state: str | None = None  # ISO3166-2-lvl4

    def to_address(self) -> str:
        """Return comma-joined sub-state fields: city, suburb, county, etc.

        Excludes state/province, country, country_code, iso_state — those are
        separate columns in ``to_dict()``.

        Example: ``"Gebang, Tulungagung"``
        """
        _EXCLUDED = {"state", "province", "region", "state_district", "country", "country_code", "iso_state"}
        parts = [
            getattr(self, f.name)
            for f in fields(self)
            if f.name not in _EXCLUDED and getattr(self, f.name) is not None
        ]
        return ", ".join(parts)

    def to_dict(self) -> dict[str, str | None]:
        """Return flat columns for manifest records.

        Columns: address (sub-state), state/province, country, country_code.
        """
        return {
            "address": self.to_address(),
            "state/province": _strip_suffix(self.state or self.province or self.region or self.state_district),
            "country": self.country,
            "country_code": self.country_code,
        }

    @classmethod
    def from_coordinate(cls, lat: float, lon: float) -> Place | None:
        """Convenience classmethod to reverse geocode a coordinate directly."""
        lat, lon = validate_coordinate(lat, lon)
        addr = reverse_geocode(lat, lon)
        if addr is None:
            return None
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in addr.items() if k in known})


@functools.lru_cache(maxsize=256)
def reverse_geocode(lat: float, lon: float) -> dict | None:
    """Return raw Nominatim address dict for the given WGS84 coordinate.

    Coordinates are rounded to 2 decimal places (~1 km) before caching.
    Returns None on failure — never raises.

    Args:
        lat: Latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees.
    """
    lat = round(lat, 2)
    lon = round(lon, 2)
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="geosave-engine")
        location = geolocator.reverse((lat, lon), timeout=5, language="en")
        if location is None:
            return None
        return location.raw.get("address", {})
    except Exception:
        return None
