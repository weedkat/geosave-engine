"""CF semantics GeoSave writes itself, beyond what odc-geo and rioxarray produce."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import Field

from geosave_engine.geodata.attrs.model import MUST_AGREE, AttrsModel

if TYPE_CHECKING:
    from collections.abc import Mapping


# CFVariable and CFCoordinate type standard_name and units alike, written once here.
type CFPhrase = Annotated[str, Field(min_length=1)] | None


# Named reducers xarray collapses with, mapped to their CF Appendix E cell method.
CELL_METHODS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "first": "point",
        "last": "point",
        "min": "minimum",
        "max": "maximum",
        "mean": "mean",
        "median": "median",
        "sum": "sum",
        "std": "standard_deviation",
        "var": "variance",
    }
)


class CFVariable(AttrsModel):
    """Describe the CF semantics of one data variable.

    What the pixels mean, not where they sit: the grid owns the coordinate
    axes and the grid mapping, and `CFCoordinate` describes those.

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

    standard_name: Annotated[CFPhrase, MUST_AGREE] = None
    long_name: CFPhrase = None
    units: Annotated[CFPhrase, MUST_AGREE] = None
    cell_methods: Annotated[CFPhrase, MUST_AGREE] = None


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
        >>> ds.gs.attrs.coords["y"].get(CFCoordinate).axis
        'Y'
    """

    standard_name: CFPhrase = None
    units: CFPhrase = None
    axis: Literal["X", "Y", "Z", "T"] | None = None
    bounds: CFPhrase = None
