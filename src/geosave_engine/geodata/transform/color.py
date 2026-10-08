"""Bake the colours a band's legend names into display channels."""

from __future__ import annotations

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.conventions import BAND_DIMENSION, SPATIAL_DIMENSIONS
from geosave_engine.geodata.utils.color import parse_color


def colorize(band: xr.DataArray) -> xr.DataArray:
    """Colour each pixel by the class its code names.

    Args:
        band: Class codes carrying a `Legend`.

    Returns:
        Georeferenced array shaped `(*axes, band, y, x)` valued in `[0, 1]`,
        whose `band` coordinate is `("red", "green", "blue")`. A pixel no
        class names is absent on every channel.

    Raises:
        ValueError: The band lists no classes, or names a class carrying no
            colour.

    Examples:
        >>> colorize(ds["landcover"]).sizes["band"]
        3
    """
    from geosave_engine.geodata.core.array import array

    legend = attrs.Legend.from_attrs(band.attrs)
    class_map = None if legend is None else legend.class_map
    if legend is None or class_map is None:
        raise ValueError(
            "band lists no classes, so its values name none to colour; write "
            "a Legend, or compose channels with GeoRaster.to_array"
        )

    colour_of = legend.color_map or {}
    codes = sorted(class_map)
    missing_colour = [code for code in codes if code not in colour_of]
    if missing_colour:
        raise ValueError(
            f"classes {missing_colour} carry no colour; give Legend.color_map an "
            f"entry for every class the band lists"
        )

    palette = np.array(
        [parse_color(colour_of[code]) for code in codes], dtype="float32"
    )
    palette /= 255.0  # (class, 3)

    pixels = band.values
    names_class = np.isin(pixels, codes)
    code_index = np.where(names_class, np.searchsorted(codes, pixels), 0)
    channels = np.where(names_class[..., None], palette[code_index], np.nan)

    axes = band.gs.axes
    return array(
        np.moveaxis(channels, -1, -3),  # (*axes, band, y, x)
        band.odc.geobox,
        dims=(*axes, BAND_DIMENSION, *SPATIAL_DIMENSIONS),
        nodata=None,  # absence is NaN here, which no fill value stands for
        coords={
            **{name: labels for name, labels in axes.items() if labels is not None},
            BAND_DIMENSION: ["red", "green", "blue"],
        },
    )
