"""Fixed identifiers for STAC endpoints, collections, and filter keys.

This module is the single source of truth for external string identifiers used
across `geodata/`.  Other modules should import from here rather than hardcoding
strings, so that a change to an upstream ID is a one-file edit.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# STAC catalog endpoints
# ---------------------------------------------------------------------------

CDSE_STAC_URL         = "https://stac.dataspace.copernicus.eu/v1"
EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1"
PLANETARY_STAC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# STAC collection identifiers
# ---------------------------------------------------------------------------

SENTINEL2_L1C = "sentinel-2-l1c"
SENTINEL2_L2A = "sentinel-2-l2a"


# ---------------------------------------------------------------------------
# CQL2 filter keys (STAC item property names used in CQL2 expressions)
# ---------------------------------------------------------------------------

class CQL2Key:
    """Namespaced property names usable in CQL2 filter expressions."""

    CLOUD_COVER         = "eo:cloud_cover"
    NODATA_PIXEL_PCT    = "s2:nodata_pixel_percentage"
    PLATFORM            = "platform"
    ORBIT_STATE         = "sat:orbit_state"
    PROCESSING_BASELINE = "s2:processing_baseline"
