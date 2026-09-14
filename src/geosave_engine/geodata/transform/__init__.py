"""Turn raw loads into model-ready layers.

Only the operations `odc.stac.load` and xarray leave undone. Nothing is
aligned, promoted, reprojected, or resampled implicitly.

Examples:
    One preprocess, end to end::

        clear = nodata.mask(raw, raw.scl.isin(CLEAR_CLASSES))
        filled = composite.mosaic([clear, last_week])
        monthly = composite.reduce(composite.resample(filled, "MS"), "median")
        dem = warp.reproject(srtm, monthly, resampling="bilinear")
        samples = tiling.Tiles([monthly], (256, 256), overlap=32)
"""

from . import composite, concat, nodata, packing, tiling, time, warp

__all__ = [
    "composite",
    "concat",
    "nodata",
    "packing",
    "tiling",
    "time",
    "warp",
]
