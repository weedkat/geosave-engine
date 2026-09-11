"""Draw a placed DataArray as a HoloViews element. Internal to `GeoRaster.plot`.

No interactive backend: every element renders through matplotlib, so
`hv.save(element, "scene.png")` writes a static figure directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import hvplot.xarray  # noqa: F401  — registers the .hvplot accessor
import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr

from geosave_engine.geodata.core.array import BAND_DIMENSION
from matplotlib.colors import ListedColormap

from geosave_engine.utils.colorize import parse_color

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from geosave_engine.utils.colorize import Palette

hv.extension("matplotlib")


def plot(
    array: xr.DataArray,
    *,
    cmap: str | Sequence[str] | None = None,
    clim: tuple[float, float] | None = None,
    class_map: Mapping[int, str] | None = None,
    color_map: Palette | None = None,
    cols: int = 4,
    title: str | None = None,
    xlabel: str | None = None,
) -> hv.Element | hv.Layout:
    """Draw a placed array the way its shape and the given palette describe it.

    A `band` dim of three draws through `rgb`; a given `class_map` draws
    through `classes`; else `continuous`. Never reads attrs itself —
    `GeoRaster.plot`/`DataArray.plot` read a `Legend` and unpack it here.

    Args:
        array: Placed band, or the three-channel stack
            `Dataset.to_array(rgb=True)` built.
        cmap: Colormap. Refused for a composite and alongside `class_map`,
            both of which carry colour rather than a quantity.
        clim: Colour limits for a band, the composite's stretch onto
            `[0, 1]` for three. Refused alongside `class_map`.
        class_map: Pixel value mapped to class name, drawing through
            `classes`. None draws the band plain.
        color_map: Pixel value mapped to hex or RGB, covering every class in
            `class_map`. Ignored without `class_map`.
        cols: Columns a `time` axis lays panels into. Ignored otherwise.
        title: Panel title, above the axes. None leaves hvplot's own.
        xlabel: Caption below the axes, e.g. a place name. None leaves
            hvplot's own.

    Returns:
        Image or RGB element over the array's grid, or a Layout of one panel
        per `time` value, `cols` wide.

    Raises:
        ValueError: `cmap` or `clim` accompanies a composite or `class_map`.

    Examples:
        >>> plot(ds.gs.to_array(["ndvi"]), cmap="RdYlGn")
        >>> plot(ds.gs.to_array(rgb=True), clim=(0.0, 0.3))
    """
    if BAND_DIMENSION in array.dims:
        if array.sizes[BAND_DIMENSION] == 3:
            if cmap is not None:
                raise ValueError(
                    "a three-channel band dim composes an RGB stack, which "
                    "carries colour rather than a quantity; drop cmap"
                )
            return rgb(array, stretch=clim, cols=cols, title=title, xlabel=xlabel)
        array = array.squeeze(BAND_DIMENSION, drop=True)

    if class_map is not None:
        overridden = [n for n, v in (("cmap", cmap), ("clim", clim)) if v is not None]
        if overridden:
            raise ValueError(
                f"class_map is given, so {overridden} cannot apply; its "
                f"palette is color_map"
            )
        return classes(
            array,
            class_map=class_map,
            color_map=color_map or {},
            cols=cols,
            title=title,
            xlabel=xlabel,
        )

    return continuous(
        array, cmap=cmap, clim=clim, cols=cols, title=title, xlabel=xlabel
    )


def continuous(
    array: xr.DataArray,
    *,
    cmap: str | Sequence[str] | None = None,
    clim: tuple[float, float] | None = None,
    cols: int = 4,
    title: str | None = None,
    xlabel: str | None = None,
) -> hv.Element | hv.Layout:
    """Draw one band's values as a continuous Image.

    Draws the values as they stand, applying no packing and reading no class
    map. Pass a label band to `classes` to draw it through a named palette.

    Args:
        array: Placed band, spanning the spatial pair and, optionally,
            `time`.
        cmap: Colormap. None leaves hvplot's own.
        clim: Colour limits. None reads them from the pixels.
        cols: Columns `time` lays panels into. Ignored without it.
        title: Panel title, above the axes. None leaves hvplot's own.
        xlabel: Caption below the axes, e.g. a place name. None leaves
            hvplot's own.

    Returns:
        Image over the band's grid, or a Layout of one panel per `time`
        value, `cols` wide.

    Examples:
        >>> continuous(ds["ndvi"], cmap="RdYlGn")
        >>> continuous(monthly["ndvi"], cols=6)  # one panel per month
    """
    y, x = array.odc.geobox.dimensions
    has_time = "time" in array.dims
    drawn = array.hvplot.image(
        x=x,
        y=y,
        groupby=["time"] if has_time else None,
        dynamic=False,  # materialise every panel; there is no live kernel
        cmap=cmap,
        clim=clim,
        title=title,
        xlabel=xlabel,
    )
    return drawn.layout(["time"]).cols(cols) if has_time else drawn


def rgb(
    array: xr.DataArray,
    *,
    stretch: tuple[float, float] | None = None,
    cols: int = 4,
    title: str | None = None,
    xlabel: str | None = None,
) -> hv.Element | hv.Layout:
    """Stretch and draw a three-channel composite as an RGB element.

    Maps `stretch` onto the unit interval, clipping outside it, reading the
    channels as `Dataset.to_array(rgb=True)` stacked them — no packing,
    no colour of their own yet.

    Args:
        array: Placed array spanning the spatial pair, `band`, and,
            optionally, `time`. `band` holds exactly three channels ordered
            red, green, blue.
        stretch: Bounds mapped onto `[0, 1]`, in the channels' own stored
            values. None takes the 2-98 percentile across the three
            channels, leaving `uint8` and values already within `[0, 1]`
            untouched.
        cols: Columns `time` lays panels into. Ignored without it.
        title: Panel title, above the axes. None leaves hvplot's own.
        xlabel: Caption below the axes, e.g. a place name. None leaves
            hvplot's own.

    Returns:
        RGB element over the array's grid, or a Layout of one panel per
        `time` value, `cols` wide.

    Raises:
        ValueError: `stretch` is empty or inverted, or it is None while
            every channel pixel is nan or the channels are flat.

    Examples:
        >>> rgb(ds.gs.to_array(rgb=True), stretch=(0.0, 0.3))
        >>> rgb(monthly.gs.to_array(rgb=True), cols=6)
    """
    y, x = array.odc.geobox.dimensions
    has_time = "time" in array.dims

    if stretch is None:
        if array.dtype == np.uint8:
            values = array.values.astype("float32") / 255.0
        else:
            finite = array.values[np.isfinite(array.values)]
            if not finite.size:
                raise ValueError("every channel pixel is nan; nothing to draw")
            if finite.min() >= 0.0 and finite.max() <= 1.0:
                values = array.values.astype("float32")
            else:
                low, high = (float(v) for v in np.percentile(finite, (2, 98)))
                if not high > low:
                    raise ValueError(
                        f"channels are flat near {low}; pass stretch=(low, high)"
                    )
                scaled = (array.values.astype("float32") - low) / (high - low)
                values = np.clip(scaled, 0.0, 1.0)
    else:
        low, high = stretch
        if not high > low:
            raise ValueError(f"stretch bounds {stretch} are empty or inverted")
        scaled = (array.values.astype("float32") - low) / (high - low)
        values = np.clip(scaled, 0.0, 1.0)
    stretched = array.copy(data=values)

    # hvplot's rgb ignores groupby, so HoloViews cuts the panels instead.
    channels = ["R", "G", "B"]
    frame = xr.Dataset(
        {
            name: stretched.isel(band=index, drop=True)
            for index, name in enumerate(channels)
        }
    )
    groupby = ["time"] if has_time else []
    drawn = hv.Dataset(frame, kdims=[x, y, *groupby], vdims=channels).to(
        hv.RGB, [x, y], channels, groupby or None
    )
    overrides = {
        n: v for n, v in (("title", title), ("xlabel", xlabel)) if v is not None
    }
    if overrides:
        drawn = drawn.opts(hv.opts.RGB(**overrides))
    return drawn.layout(["time"]).cols(cols) if has_time else drawn


def classes(
    array: xr.DataArray,
    *,
    class_map: Mapping[int, str],
    color_map: Palette,
    cols: int = 4,
    title: str | None = None,
    xlabel: str | None = None,
) -> hv.Element | hv.Layout:
    """Draw a label band through its palette, named on the colorbar.

    Pixels draw on their class's position rather than its code, so a sparsely
    numbered product leaves no empty colour band and an unnamed pixel draws as
    none. Colour limits follow the class count, so they are not the caller's.

    Args:
        array: Placed band holding class codes, spanning the spatial pair
            and, optionally, `time`.
        class_map: Pixel value mapped to class name.
        color_map: Pixel value mapped to hex or RGB, covering every class.
        cols: Columns `time` lays panels into. Ignored without it.
        title: Panel title, above the axes. None leaves hvplot's own.
        xlabel: Caption below the axes, e.g. a place name. None leaves
            hvplot's own.

    Returns:
        Image drawing each class in its own colour and naming it on the
        colorbar, or a Layout of one panel per `time` value, `cols` wide.

    Raises:
        ValueError: A named class carries no colour.

    Examples:
        >>> classes(ds["cover"], class_map={0: "water"}, color_map={0: "#419bdf"})
        >>> classes(yearly["cover"], cols=6, **declared)
    """
    y, x = array.odc.geobox.dimensions
    has_time = "time" in array.dims

    uncoloured = sorted(code for code in class_map if code not in color_map)
    if uncoloured:
        raise ValueError(
            f"classes {uncoloured} carry no colour; give color_map an entry "
            f"for every class in class_map"
        )

    codes = sorted(class_map)
    hexes = ["#%02x%02x%02x" % parse_color(color_map[code]) for code in codes]
    ticks = [(position, class_map[code]) for position, code in enumerate(codes)]

    named = np.isin(array.values, codes)
    positions = np.where(named, np.searchsorted(codes, array.values), np.nan)
    drawn = array.copy(data=positions).hvplot.image(
        x=x,
        y=y,
        groupby=["time"] if has_time else None,
        dynamic=False,  # materialise every panel; there is no live kernel
        clim=(-0.5, len(codes) - 0.5),
        colorbar=True,
        xlabel=xlabel,
        title=title,
    )
    # A bare list of colours breaks matplotlib's GridSpace.
    styled = drawn.opts(
        hv.opts.Image(
            cmap=ListedColormap(hexes), color_levels=len(codes), cbar_ticks=ticks
        )
    )
    return styled.layout(["time"]).cols(cols) if has_time else styled
