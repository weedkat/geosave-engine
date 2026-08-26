"""ArraySpec: the array identity raw pixels lose. See ArraySpec for details."""
from __future__ import annotations

from datetime import datetime as dt  # runtime import — pydantic resolves field annotations
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray

from geosave_engine.geodata.extensions.base import GeoExtension

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr


class ArraySpec(GeoExtension):
    """Band names, timestamps and fill value — what a bare NumPy array loses.

    Restamped off `data` every time pixels are attached, so the array stays
    authoritative. It is load-bearing only where there are no pixels to read:
    an anchor decoded out of a store, or one handed to a `ContextFn`.

    Args:
        bands: Band names in coordinate order.
        times: Observation datetimes, or None for a timeless array.
        nodata: Fill sentinel, or None when the array declares none.

    Examples:
        >>> tile.anchor.header.array.bands
        ('B04', 'B08')
    """

    NAMESPACE: ClassVar[str] = "array"
    SETTABLE: ClassVar[bool] = False

    bands: tuple[str, ...] | None = None
    times: tuple[dt, ...] | None = None
    nodata: float | int | None = None

    def reconcile(self, data: xr.DataArray) -> Self:
        """Re-read this namespace off the array, discarding what was stored.

        The pixels own these three facts, so a stored value that disagrees
        with them loses. `_SpatialArray` seeds a bare `ArraySpec()` on every
        construction, which is what makes this the value on the header.

        Args:
            data: This array's own canonical pixel data.

        Returns:
            Bands off the `band` coordinate, times off the `time`
            coordinate when there is one, nodata off the array's own
            declaration.
        """
        times = (
            tuple(value.astype("datetime64[us]").item() for value in data.time.values)
            if "time" in data.dims
            else None
        )
        nodata = data.rio.nodata
        return type(self)(
            bands=tuple(str(value) for value in data.band.values),
            times=times,
            nodata=nodata.item() if isinstance(nodata, np.generic) else nodata,
        )

    @classmethod
    def combine(cls, values: Sequence[GeoExtension | None]) -> None:
        """Never carry this namespace onto a composed array.

        `concat` changes the very bands and times this records, and the
        composed result restamps its own.

        Args:
            values: This namespace's value from each array being composed.

        Returns:
            None, always.
        """
        return None
