"""Reverse geocoding backed by OpenStreetMap Nominatim."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, fields

from .crs import validate_wgs84_coordinate

_ADMIN_SUFFIXES = re.compile(
    r"\s+(Province|State|Region|District|Prefecture|Department|Governorate|Oblast|Krai|County|Municipality)$",
    re.IGNORECASE,
)


def _strip_administrative_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    return _ADMIN_SUFFIXES.sub("", value).strip() or None


@dataclass(frozen=True)
class Place:
    """Store normalized Nominatim address components.

    Fields retain Nominatim's administrative levels so callers can choose
    their own display or storage representation.
    """

    city: str | None = None
    town: str | None = None
    village: str | None = None
    hamlet: str | None = None
    isolated_dwelling: str | None = None
    suburb: str | None = None
    borough: str | None = None
    quarter: str | None = None
    neighbourhood: str | None = None
    city_district: str | None = None
    county: str | None = None
    municipality: str | None = None
    district: str | None = None
    state_district: str | None = None
    state: str | None = None
    province: str | None = None
    region: str | None = None
    country: str | None = None
    country_code: str | None = None
    iso_state: str | None = None

    def to_address(self) -> str:
        """Build a human-readable address below the state level.

        Returns:
            Comma-separated populated-place and district components.
        """
        excluded = {
            "state",
            "province",
            "region",
            "state_district",
            "country",
            "country_code",
            "iso_state",
        }
        parts = [
            value
            for field in fields(self)
            if field.name not in excluded
            and (value := getattr(self, field.name)) is not None
        ]
        return ", ".join(parts)

    def to_dict(self) -> dict[str, str | None]:
        """Build flat location fields for records and manifests.

        Returns:
            Address, state or province, country, and country code.
        """
        return {
            "address": self.to_address(),
            "state/province": _strip_administrative_suffix(
                self.state or self.province or self.region or self.state_district
            ),
            "country": self.country,
            "country_code": self.country_code,
        }

    @classmethod
    def from_coordinate(cls, latitude: float, longitude: float) -> Place | None:
        """Reverse geocode one WGS84 coordinate.

        Args:
            latitude: Latitude in WGS84 degrees.
            longitude: Longitude in WGS84 degrees.

        Returns:
            Resolved place, or None when the service has no result or cannot
            be reached.

        Raises:
            ValueError: Latitude is outside [-90, 90].
        """
        address = reverse_geocode(latitude, longitude)
        if address is None:
            return None
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in address.items() if key in known})


def reverse_geocode(latitude: float, longitude: float) -> dict[str, str] | None:
    """Return Nominatim address fields for one WGS84 coordinate.

    Results share a cache at two decimal places, approximately a one-kilometre
    grid. Network and service failures return None.

    Args:
        latitude: Latitude in WGS84 degrees.
        longitude: Longitude in WGS84 degrees.

    Returns:
        Raw address fields, or None when no result is available.

    Raises:
        ValueError: Latitude is outside [-90, 90].
    """
    latitude, longitude = validate_wgs84_coordinate(latitude, longitude)
    return _request_nominatim_address(round(latitude, 2), round(longitude, 2))


@functools.lru_cache(maxsize=256)
def _request_nominatim_address(
    latitude: float,
    longitude: float,
) -> dict[str, str] | None:
    from geopy.exc import GeopyError
    from geopy.geocoders import Nominatim

    try:
        client = Nominatim(user_agent="geosave-engine")
        location = client.reverse((latitude, longitude), timeout=5, language="en")
    except GeopyError:
        return None
    if location is None:
        return None
    address = location.raw.get("address", {})
    return {str(key): str(value) for key, value in address.items()}
