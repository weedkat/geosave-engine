"""RenderHints: which bands carry which display role. See RenderHints for details."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar, Self

from geosave_engine.geodata.extensions.base import GeoExtension

if TYPE_CHECKING:
    import xarray as xr


class RenderHints(GeoExtension):
    """Which bands carry which display role.

    Keyed to band identity, so it is dropped when the bands it names go
    away. Pixel-value meaning lives in `Legend`, which is keyed to values
    and survives band changes.

    Args:
        rgb_bands: Which three bands render as red/green/blue, by band-coord name.

    Examples:
        >>> raster.rebase(render={"rgb_bands": ("B04", "B03", "B02")})
    """

    NAMESPACE: ClassVar[str] = "render"

    rgb_bands: tuple[str, str, str] | None = None

    def reconcile(self, data: xr.DataArray) -> Self | None:
        """Drop this namespace if `rgb_bands` names a band the array doesn't have.

        Args:
            data: This array's own canonical pixel data.

        Returns:
            Self unchanged when every named band exists, or when
            `rgb_bands` isn't set; None to drop this namespace otherwise.
        """
        if self.rgb_bands is None:
            return self
        band_names = {str(value) for value in data.band.values}
        # subset check: every named band must actually exist, not just match in count
        if set(self.rgb_bands) <= band_names:
            return self
        missing = sorted(set(self.rgb_bands) - band_names)
        warnings.warn(
            f"render.rgb_bands references missing bands {missing}; available bands: "
            f"{sorted(band_names)}. Dropping the band roles; any legend is unaffected"
        )
        return None
