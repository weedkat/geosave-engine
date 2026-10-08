"""Packing that decodes stored digital numbers into physical values."""

from __future__ import annotations

from typing import Annotated

from geosave_engine.geodata.attrs.model import MUST_AGREE, AttrsModel


class Packing(AttrsModel):
    """Describe how one variable's stored values decode to physical values.

    Rasters stay in their stored digital numbers while in memory, so a variable
    carrying these fields still holds what its source published. Absence is
    `Nodata`, which packing never touches.

    Args:
        scale_factor: Multiplier applied to each stored value.
        add_offset: Value added after scaling.

    Examples:
        >>> ds.gs.rebase(Packing(scale_factor=1e-4, add_offset=-0.1), target="B04")
        >>> ds.gs.attrs.data_vars["B04"].get(Packing)
        Packing(scale_factor=0.0001, add_offset=-0.1)
    """

    scale_factor: Annotated[float | None, MUST_AGREE] = None
    add_offset: Annotated[float | None, MUST_AGREE] = None
