"""Shared geodata exception and warning types."""


class AnchorFetchError(RuntimeError):
    """Raised when a source has no usable data for an anchor."""


class TileDownloadError(RuntimeError):
    """Base for a download() failure — catch this to skip an unusable tile without retrying blind."""


class TileDecodeError(TileDownloadError, OSError):
    """GDAL logged a tile decode failure (truncated/corrupt byte-range read) — transient, safe to retry."""


class GeoSaveWarning(UserWarning):
    """Base for data GeoSave accepted despite it lacking something.

    Escalate the whole family to errors with
    `warnings.simplefilter("error", GeoSaveWarning)`.
    """


class MissingCRSWarning(GeoSaveWarning):
    """Opened raster declares no CRS, so ground-referenced operations will raise."""


class UnreferencedGridWarning(GeoSaveWarning):
    """Opened raster is indexed in pixels, carrying no transform onto any ground."""


class DroppedAttrsWarning(GeoSaveWarning):
    """Joined rasters did not state an attr alike, so the result states nothing."""


class DroppedBucketsWarning(GeoSaveWarning):
    """A resample bucket covered no observation, so it was dropped from the axis."""


class UnmatchedBucketsWarning(GeoSaveWarning):
    """A broadcast source bucket matched no target label, so it was left out of the result."""
