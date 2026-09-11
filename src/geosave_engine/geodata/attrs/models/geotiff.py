"""Baseline TIFF tags GDAL carries as dataset-level metadata."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar, Literal, Self

import numpy as np
from pydantic import field_serializer, field_validator

from geosave_engine.geodata.attrs.model import AttrsModel

DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


class GeoTIFFTags(AttrsModel):
    """State the baseline TIFF tags a GeoTIFF carries.

    Fields are named for the tags themselves, so what the model states is what
    GDAL writes. `TIFFTAG_MINSAMPLEVALUE` and `TIFFTAG_MAXSAMPLEVALUE` are
    absent because GDAL reports them read-only.

    Args:
        TIFFTAG_DOCUMENTNAME: Name of the document the image came from.
        TIFFTAG_IMAGEDESCRIPTION: Free-text description of the image.
        TIFFTAG_SOFTWARE: Software that produced the file.
        TIFFTAG_DATETIME: When the image was made. Written as GDAL's
            ``YYYY:mm:dd HH:MM:SS``, which holds whole seconds only.
        TIFFTAG_ARTIST: Person who made the image.
        TIFFTAG_HOSTCOMPUTER: Machine the file was written on.
        TIFFTAG_COPYRIGHT: Copyright notice.
        TIFFTAG_XRESOLUTION: Pixels per resolution unit across the image.
        TIFFTAG_YRESOLUTION: Pixels per resolution unit down the image.
        TIFFTAG_RESOLUTIONUNIT: Unit the resolutions count against: 1 for
            none, 2 for inches, 3 for centimetres.

    Examples:
        >>> ds.gs.rebase(geotiff={"TIFFTAG_ARTIST": "GeoSave"})
    """

    NAME: ClassVar[str] = "geotiff"

    TIFFTAG_DOCUMENTNAME: str | None = None
    TIFFTAG_IMAGEDESCRIPTION: str | None = None
    TIFFTAG_SOFTWARE: str | None = None
    TIFFTAG_DATETIME: datetime | None = None
    TIFFTAG_ARTIST: str | None = None
    TIFFTAG_HOSTCOMPUTER: str | None = None
    TIFFTAG_COPYRIGHT: str | None = None
    TIFFTAG_XRESOLUTION: float | None = None
    TIFFTAG_YRESOLUTION: float | None = None
    TIFFTAG_RESOLUTIONUNIT: Literal[1, 2, 3] | None = None

    @field_validator("TIFFTAG_DATETIME", mode="before")
    @classmethod
    def _parse_datetime(cls, value: object) -> datetime | None:
        """Read the instant, however the caller or the file spells it.

        Args:
            value: A datetime, a numpy instant, or GDAL's own tag text.

        Returns:
            The instant, or None where the tag is absent.

        Raises:
            ValueError: The text is not GDAL's format, the value is no instant
                at all, or it carries sub-second precision the tag cannot hold.
        """
        if isinstance(value, str):
            try:
                value = datetime.strptime(value, DATETIME_FORMAT)  # noqa: DTZ007
            except ValueError:
                raise ValueError(
                    f"{value!r} is not the 'YYYY:mm:dd HH:MM:SS' GDAL writes"
                ) from None
        elif isinstance(value, np.datetime64):
            value = value.astype("datetime64[us]").astype(datetime)

        if value is not None and not isinstance(value, datetime):
            raise ValueError(
                f"a datetime tag needs an instant, got {type(value).__name__}"
            )
        if value is not None and value.microsecond:
            raise ValueError(
                f"{value} carries sub-second precision, which the tag cannot "
                f"hold; round it to whole seconds first"
            )
        return value

    @field_validator("TIFFTAG_RESOLUTIONUNIT", mode="before")
    @classmethod
    def _parse_resolution_unit(cls, value: object) -> object:
        """Read the unit, which GDAL spells out on the way back.

        GDAL reports this tag as ``"2 (pixels/inch)"`` rather than ``2``, so
        the count is read off the front.

        Args:
            value: The unit, as a number or as GDAL's own text.

        Returns:
            The unit as a number, or whatever came in for pydantic to refuse.
        """
        if not isinstance(value, str):
            return value
        count = value.split(maxsplit=1)[0] if value.strip() else value
        return int(count) if count.isdigit() else value

    @field_serializer("TIFFTAG_DATETIME")
    def _format_datetime(self, value: datetime | None) -> str | None:
        """Spell the instant the way GDAL does.

        Args:
            value: The instant this model states.

        Returns:
            Tag text shaped ``YYYY:mm:dd HH:MM:SS``, or None where the tag is
            absent.
        """
        return None if value is None else value.strftime(DATETIME_FORMAT)

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine TIFF tags, dropping disagreements.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side did not state it.

        Returns:
            Model carrying the fields every side states alike, and the
            attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: `models` is empty.
        """
        return cls._combine_fields(sides, must_agree=())
