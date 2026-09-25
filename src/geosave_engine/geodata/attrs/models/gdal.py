"""What a GDAL band says about the variable it holds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import Field, field_validator
from rasterio.enums import ColorInterp

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.geodata.attrs.namespace import AttrsNamespace

if TYPE_CHECKING:
    import xarray as xr

# GDAL's names for the three display channels, in the order a composite draws them.
_RGB = (ColorInterp.red.name, ColorInterp.green.name, ColorInterp.blue.name)


class GDALVariable(AttrsModel):
    """Name the variable one raster band holds.

    GDAL's band description carries CF's `long_name`, which describes the
    pixels rather than identifying them, so a band carries its variable name as
    its own metadata item instead.

    Args:
        variable_name: Name of the variable this band holds.
        colorinterp: How a reader should interpret the band, named as GDAL
            names it, such as `"red"`, `"alpha"`, or `"palette"`. GDAL's
            `"undefined"` interprets nothing and reads as None.

    Raises:
        ValueError: `colorinterp` names no interpretation GDAL knows.

    Examples:
        >>> ds.gs.rebase(GDALVariable(variable_name="B04", colorinterp="red"))
        >>> ds.gs.attrs.data_vars["B04"].get(GDALVariable)
        GDALVariable(variable_name='B04', colorinterp='red')
    """

    NAME: ClassVar[str] = "gdal"

    variable_name: Annotated[str, Field(min_length=1)] | None = None
    colorinterp: str | None = None
    description: Annotated[str, Field(min_length=1)] | None = None

    @field_validator("colorinterp", mode="before")
    @classmethod
    def _read_interpretation(cls, value: object) -> object:
        """Read the interpretation, however rasterio or the file spells it.

        Args:
            value: A `ColorInterp` member or the name of one.

        Returns:
            The member's name, or None where it is GDAL's `"undefined"`.

        Raises:
            ValueError: The name belongs to no `ColorInterp` member.
        """
        if isinstance(value, ColorInterp):
            value = value.name
        if value == ColorInterp.undefined.name:
            return None
        if isinstance(value, str) and value not in ColorInterp.__members__:
            raise ValueError(
                f"{value!r} names no GDAL colour interpretation; use one of "
                f"{sorted(ColorInterp.__members__)}"
            )
        return value

    @classmethod
    def rgb_indices(cls, ds: xr.Dataset) -> tuple[int, int, int]:
        """Find the bands a true-colour composite draws, red first.

        Colour interpretation says what a band measures, so only a raster
        measuring red, green, and blue composes itself. Any other composite,
        false colour among them, names its three bands at the call site.

        Args:
            ds: Raster whose bands are its data variables, in band order.

        Returns:
            Position of the red, the green, and the blue band, indexing `ds`'s
            data variables.

        Raises:
            ValueError: A colour is measured by no band. Two bands measuring
                one colour compose from the first.

        Examples:
            >>> GDALVariable.rgb_indices(scene)
            (0, 1, 2)
        """
        bands = (
            AttrsNamespace.from_attrs(variable.attrs).get(cls)
            for variable in ds.data_vars.values()
        )
        measures = [None if band is None else band.colorinterp for band in bands]

        absent = [colour for colour in _RGB if colour not in measures]
        if absent:
            raise ValueError(
                f"no band of {list(ds.data_vars)} draws {absent}, so this raster "
                f"composes no colour of its own; name its channels once with "
                f"ds.gs.write_rgb('B04', 'B03', 'B02'), or name three bands for "
                f"one drawing with ds.gs.plot(('B08', 'B04', 'B03'))"
            )
        red, green, blue = (measures.index(colour) for colour in _RGB)
        return red, green, blue
