"""CF semantics GeoSave writes itself, beyond what odc-geo and rioxarray produce."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

from pydantic import Field

from geosave_engine.geodata.attrs.model import AttrsModel

from .stac import shared_asset_fields

if TYPE_CHECKING:
    import pystac
    from odc.geo.geobox import GeoBox

# CFVariable and CFCoordinate declare standard_name and units alike, stated once here.
type CFPhrase = Annotated[str, Field(min_length=1)] | None


class CFVariable(AttrsModel):
    """Describe the CF semantics of one data variable.

    Structural grid metadata is not represented here. Profile owns coordinate
    axes and grid mappings, while this model carries semantics supplied by a
    source or caller.

    Args:
        standard_name: CF standard name, e.g. `"toa_bidirectional_reflectance"`.
        long_name: Human-readable variable name.
        units: UDUNITS string, e.g. `"1"` for reflectance.
        cell_methods: CF cell-methods phrase a time resample writes, e.g.
            `"time: mean"`.

    Examples:
        >>> ds.gs.rebase(
        ...     CFVariable(standard_name="surface_albedo", units="1"), target="B04"
        ... )
    """

    NAME: ClassVar[str] = "cf"

    standard_name: CFPhrase = None
    long_name: CFPhrase = None
    units: CFPhrase = None
    cell_methods: CFPhrase = None

    @classmethod
    def from_stac_asset(cls, items: Sequence[pystac.Item], asset: str) -> Self:
        """Read what every item of one load says alike about an asset.

        These fields label the pixels for a reader rather than deciding what
        they mean, so a field the items describe differently is left out and
        the load stands.

        Args:
            items: Items making up one load.
            asset: Asset name to read, which names the variable it loads into.

        Returns:
            Model carrying the semantics every item states identically, its
            other fields unset.

        Examples:
            >>> CFVariable.from_stac_asset(matched, "B04").units
            '1'
        """
        return cls.model_validate(
            shared_asset_fields(
                items,
                asset,
                {"unit": "units", "description": "long_name"},
                on_conflict="drop",
            )
        )

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine CF variable semantics.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side did not state it.

        Returns:
            Combined model, and the attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: A semantic field disagrees or `models` is empty.
        """
        return cls._combine_fields(
            sides, must_agree=("standard_name", "units", "cell_methods")
        )


class CFCoordinate(AttrsModel):
    """Describe the CF semantics of one coordinate.

    The grid stays the authority on where pixels sit. These fields say what
    the coordinate measures, so a CF reader can identify the axis.

    Args:
        standard_name: CF standard name, e.g. `"projection_y_coordinate"`.
        units: UDUNITS string, e.g. `"metre"` or `"degrees_north"`.
        axis: CF axis letter the coordinate spans.
        bounds: Name of the variable holding this coordinate's cell edges,
            e.g. `"time_bnds"`.

    Examples:
        >>> ds.gs.attrs.get(CFCoordinate, target="y").axis
        'Y'
    """

    NAME: ClassVar[str] = "coordinate"

    standard_name: CFPhrase = None
    units: CFPhrase = None
    axis: Literal["X", "Y", "Z", "T"] | None = None
    bounds: CFPhrase = None

    @classmethod
    def from_geobox(cls, geobox: GeoBox) -> dict[str, Self]:
        """Read what each spatial coordinate of a grid measures.

        A projected grid measures `projection_y_coordinate` and
        `projection_x_coordinate` in its own length unit; a geographic one
        measures `latitude` and `longitude` in degrees.

        Args:
            geobox: Grid carrying a CRS.

        Returns:
            {
                "<coordinate name>": semantics that axis carries,
            }
            Named as the grid names its own dimensions. Neither states
            `bounds`, because the grid's affine already gives every pixel's
            edges.

        Raises:
            ValueError: `geobox` declares no CRS, so its axes measure pixels
                rather than ground position.

        Examples:
            >>> CFCoordinate.from_geobox(utm_geobox)["y"].standard_name
            'projection_y_coordinate'
            >>> CFCoordinate.from_geobox(wgs84_geobox)["longitude"].units
            'degrees_east'
        """
        crs = geobox.crs
        if crs is None:
            raise ValueError(
                "geobox declares no CRS, so its axes measure pixels rather than "
                "ground position; assign one before describing them"
            )
        names = (
            ("latitude", "longitude")
            if crs.geographic
            else ("projection_y_coordinate", "projection_x_coordinate")
        )
        axes: tuple[Literal["X", "Y", "Z", "T"], ...] = ("Y", "X")
        return {
            dimension: cls(standard_name=name, units=unit, axis=axis)
            for dimension, name, unit, axis in zip(
                geobox.dimensions, names, crs.units, axes, strict=True
            )
        }

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine coordinate semantics, refusing disagreements.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side did not state it.

        Returns:
            Combined model, and the attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: A field disagrees or `models` is empty.
        """
        return cls._combine_fields(sides, must_agree=cls.model_fields)
