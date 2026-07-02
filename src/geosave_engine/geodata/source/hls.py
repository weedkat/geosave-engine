from __future__ import annotations

import pystac
import xarray as xr

from .base import Source


class HLSSource(Source):
    """Source for HLS (Harmonized Landsat Sentinel-2) collections.

    Available on Planetary Computer: ``hls2-s30`` (Sentinel-2 derived),
    ``hls2-l30`` (Landsat derived).
    """

    def preprocess(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        """Apply radiometric scaling to HLS bands.

        HLS S30/L30 store surface reflectance as scaled int16 (DN = reflectance × 10000).
        Unlike Sentinel-2, HLS items don't carry per-band scale/offset in STAC metadata —
        the scale factor is fixed at 0.0001 per the HLS product specification.

        Note: GraniteGeospatialBiomass expects raw DN values (not scaled).
        Apply this preprocess only when consuming HLS data for models trained on reflectance.
        """
        return ds * 0.0001
