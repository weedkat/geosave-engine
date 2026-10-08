"""Where in the spectrum one variable was measured."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from geosave_engine.geodata.attrs.model import AttrsModel


class Spectral(AttrsModel):
    """Name the spectral band one variable holds.

    Args:
        common_name: Band name shared across sensors, e.g. `"red"` or `"nir"`.
        center_wavelength: Centre of the band, in micrometres.
        full_width_half_max: Width of the band at half its peak response, in
            micrometres.

    Examples:
        >>> ds.gs.rebase(
        ...     Spectral(common_name="red", center_wavelength=0.665), target="B04"
        ... )
        >>> ds.gs.attrs.data_vars["B04"].get(Spectral).common_name
        'red'
    """

    common_name: Annotated[str, Field(min_length=1)] | None = None
    center_wavelength: Annotated[float, Field(gt=0)] | None = None
    full_width_half_max: Annotated[float, Field(gt=0)] | None = None
