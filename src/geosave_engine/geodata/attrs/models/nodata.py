"""The stored value a variable's absent pixels hold."""

from __future__ import annotations

from typing import ClassVar, Mapping

from geosave_engine.geodata.attrs.model import AttrsModel


class Nodata(AttrsModel):
    """Name the stored value marking one variable's absent pixels.

    Args:
        fill_value: Stored value marking absence. CF spells it `_FillValue`
            and odc spells it `nodata`; both are written, so the two can never
            name different pixels.

    Examples:
        >>> ds.gs.rebase(Nodata(fill_value=0), target="B04")
        >>> ds.gs.attrs.data_vars["B04"].get(Nodata)
        Nodata(fill_value=0)
    """

    NAME: ClassVar[str] = "nodata"
    field_keys: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "fill_value": ("_FillValue", "nodata")
    }

    fill_value: int | float | None = None
