"""Shared exception types for geodata source/pipeline fetch failures.

Lives at the ``geodata`` package root — both ``geodata.stac`` (raises) and
``geodata.pipeline`` (documents/catches) depend on it, and neither of those
should depend on the other just for an exception type.
"""


class AnchorFetchError(RuntimeError):
    """Raised when a source has no usable data for an anchor."""


class TileDownloadError(RuntimeError):
    """Base for a download() failure — catch this to skip an unusable tile without retrying blind."""


class TileDecodeError(TileDownloadError, OSError):
    """GDAL logged a tile decode failure (truncated/corrupt byte-range read) — transient, safe to retry."""


class UnknownExtensionError(ValueError):
    """An attr used a namespace no GeoExtension subclass has registered — import its module first."""
