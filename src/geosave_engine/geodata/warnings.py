"""Warnings for data loss, assumptions, and incomplete results."""

__all__ = [
    "GeoSaveWarning",
    "UnreadMaskWarning",
    "AssumedFillWarning",
    "DroppedAttrsWarning",
    "DroppedInstantsWarning",
    "UncoveredInstantsWarning",
    "DroppedFramesWarning",
]


class GeoSaveWarning(UserWarning):
    """Base category for GeoSave warnings.

    Escalate the family with ``warnings.simplefilter("error", GeoSaveWarning)``.
    """


class UnreadMaskWarning(GeoSaveWarning):
    """A band's absence mask is not read."""


class AssumedFillWarning(GeoSaveWarning):
    """A warped variable carried no fill value, so one was chosen."""


class DroppedAttrsWarning(GeoSaveWarning):
    """Conflicting or incomplete source metadata was dropped."""


class DroppedInstantsWarning(GeoSaveWarning):
    """Trailing instants did not fill a frame and were left out."""


class UncoveredInstantsWarning(GeoSaveWarning):
    """Instants outside the interval shared by every group name no slot."""


class DroppedFramesWarning(GeoSaveWarning):
    """Frames lacking a scene for some group were not emitted."""
