"""TIFF metadata GDAL carries at dataset level."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from odc.geo.geobox import GeoBox
from pydantic import field_serializer, field_validator

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.geodata.attrs.models.acdd import ACDD

if TYPE_CHECKING:
    import xarray as xr

DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


class GeoTIFFTags(AttrsModel):
    """State the TIFF metadata a GeoTIFF carries.

    Baseline fields use their TIFF tag names. ``GEOSAVE_DATETIME`` is precise
    GeoSave metadata stored in GDAL's metadata tag. `TIFFTAG_MINSAMPLEVALUE`
    and `TIFFTAG_MAXSAMPLEVALUE` are absent because GDAL reports them read-only.

    Args:
        TIFFTAG_DOCUMENTNAME: Name of the document the image came from.
        TIFFTAG_IMAGEDESCRIPTION: Free-text description of the image.
        TIFFTAG_SOFTWARE: Software that produced the file.
        TIFFTAG_DATETIME: When the image was made. Written as GDAL's
            ``YYYY:mm:dd HH:MM:SS``, which holds whole seconds only.
        GEOSAVE_DATETIME: The same instant as ISO 8601 text at its original
            precision. GDAL stores this non-standard metadata inside the TIFF.
        TIFFTAG_ARTIST: Person who made the image.
        TIFFTAG_HOSTCOMPUTER: Machine the file was written on.
        TIFFTAG_COPYRIGHT: Copyright notice.
        TIFFTAG_XRESOLUTION: Pixels per resolution unit across the image.
        TIFFTAG_YRESOLUTION: Pixels per resolution unit down the image.
        TIFFTAG_RESOLUTIONUNIT: Unit the resolutions count against: 1 for
            none, 2 for inches, 3 for centimetres.

    Examples:
        >>> ds.gs.rebase(geotiff_tags={"TIFFTAG_ARTIST": "GeoSave"})
    """

    TIFFTAG_DOCUMENTNAME: str | None = None
    TIFFTAG_IMAGEDESCRIPTION: str | None = None
    TIFFTAG_SOFTWARE: str | None = None
    TIFFTAG_DATETIME: datetime | None = None
    GEOSAVE_DATETIME: str | None = None
    TIFFTAG_ARTIST: str | None = None
    TIFFTAG_HOSTCOMPUTER: str | None = None
    TIFFTAG_COPYRIGHT: str | None = None
    TIFFTAG_XRESOLUTION: float | None = None
    TIFFTAG_YRESOLUTION: float | None = None
    TIFFTAG_RESOLUTIONUNIT: Literal[1, 2, 3] | None = None

    @classmethod
    def from_xarray(
        cls,
        obj: xr.Dataset | xr.DataArray,
        *,
        map_scale: float | None = None,
    ) -> Self:
        """Build the TIFF tags that describe one raster file.

        Existing TIFF tags are retained. An ACDD summary supplies the image
        description, a scalar time coordinate supplies both the standard
        whole-second TIFF datetime and GeoSave's precise datetime metadata,
        and ``map_scale`` converts ground resolution to pixels per centimetre.

        Args:
            obj: Raster represented by this file.
            map_scale: Map denominator, such as ``10_000`` for 1:10,000.

        Returns:
            Tags synchronized with the raster and requested map scale.

        Raises:
            ValueError: The scale is invalid, or physical resolution cannot be
                calculated from a regular grid in a projected CRS.
        """
        values = (cls.from_attrs(obj.attrs) or cls()).to_attrs()

        acdd = ACDD.from_attrs(obj.attrs)
        if acdd is not None and acdd.summary is not None:
            values["TIFFTAG_IMAGEDESCRIPTION"] = acdd.summary

        time = obj.coords.get("time")
        if time is not None and time.ndim == 0:
            instant = time.values
            if isinstance(instant, np.datetime64) and not np.isnat(instant):
                values["TIFFTAG_DATETIME"] = instant.astype("datetime64[s]")
                values["GEOSAVE_DATETIME"] = np.datetime_as_string(instant, unit="auto")
            else:
                values["TIFFTAG_DATETIME"] = instant
                values["GEOSAVE_DATETIME"] = None

        if map_scale is not None:
            try:
                scale = float(map_scale)
            except (TypeError, ValueError):
                raise ValueError("map_scale must be a positive finite number") from None
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("map_scale must be a positive finite number")

            grid = obj.odc.geobox
            if not isinstance(grid, GeoBox):
                raise ValueError("map_scale needs a regular grid with a projected CRS")
            if grid.crs is None or not grid.crs.projected:
                raise ValueError("map_scale needs a regular grid in a projected CRS")

            axes = grid.crs.proj.axis_info
            if len(axes) < 2:
                raise ValueError(
                    "map_scale needs a projected CRS with linear axis units"
                )
            x_factor = axes[0].unit_conversion_factor
            y_factor = axes[1].unit_conversion_factor
            if x_factor is None or y_factor is None:
                raise ValueError(
                    "map_scale needs a projected CRS with linear axis units"
                )

            x_metres = abs(grid.resolution.x) * x_factor
            y_metres = abs(grid.resolution.y) * y_factor
            if not np.isfinite(x_metres) or not np.isfinite(y_metres):
                raise ValueError("map_scale needs finite ground resolution")
            if x_metres <= 0 or y_metres <= 0:
                raise ValueError("map_scale needs positive ground resolution")

            values.update(
                TIFFTAG_XRESOLUTION=scale / (100 * x_metres),
                TIFFTAG_YRESOLUTION=scale / (100 * y_metres),
                TIFFTAG_RESOLUTIONUNIT=3,
            )

        return cls.model_validate(values)

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

    @field_validator("GEOSAVE_DATETIME")
    @classmethod
    def _validate_geosave_datetime(cls, value: str | None) -> str | None:
        """Require an ISO instant that NumPy can restore without precision loss."""
        if value is None:
            return None
        try:
            instant = np.datetime64(value)
        except ValueError:
            raise ValueError("GEOSAVE_DATETIME needs an ISO 8601 instant") from None
        if np.isnat(instant):
            raise ValueError("GEOSAVE_DATETIME needs an ISO 8601 instant")
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
            value: The instant this model carries.

        Returns:
            Tag text shaped ``YYYY:mm:dd HH:MM:SS``, or None where the tag is
            absent.
        """
        return None if value is None else value.strftime(DATETIME_FORMAT)
