"""Classification extension: the classes a band's pixel values name."""

from __future__ import annotations

import re
from typing import cast

import pystac
import xarray as xr
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.raster import RasterBand, RasterExtension

from geosave_engine.geodata.attrs import AttrsModel, Legend
from geosave_engine.geodata.utils.color import parse_color

# What the schema lets a class be named.
_CLASS_NAME = re.compile(r"[0-9A-Za-z_-]+")


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write each variable's legend as classes on its Raster band.

    Classes sit inside the band they describe, so the Raster extension must
    have written the asset's bands first.

    Args:
        asset: Asset already added to its Item, carrying its Raster bands.
        raster: Raster as `read_raster` opens it. A variable with no legend,
            or one that lists bit masks, writes nothing.

    Raises:
        ValueError: A class name carries a character the schema refuses.

    Examples:
        >>> write(asset, label)
        >>> band = RasterExtension.ext(asset).bands[0]
        >>> ClassificationExtension.ext(band).classes[1].name
        'forest'
    """
    header = raster.gs.attrs
    bands = RasterExtension.ext(asset).bands or []
    for name, band in zip(raster.gs.variables, bands, strict=True):
        legend = header.data_vars[name].get(Legend)
        # CF masks do not state the bit offset, length and names a STAC bitfield needs.
        if legend is None or legend.class_map is None or legend.flag_masks is not None:
            continue
        ClassificationExtension.ext(band).apply(classes=_classes(legend))
        # A band has no owner to declare the schema on.
        ClassificationExtension.add_to(cast("pystac.Item", asset.owner))


def read(band: RasterBand) -> list[AttrsModel]:
    """Read the legend one Raster band's classes state.

    Args:
        band: One band of an asset's Raster listing.

    Returns:
        One `Legend` naming each class and its colour hint. Empty where the
        band lists no classes.

    Examples:
        >>> (legend,) = read(RasterExtension.ext(asset).bands[0])
        >>> legend.class_map
        {0: 'background', 1: 'forest'}
    """
    classes = ClassificationExtension.ext(band).classes
    if not classes:
        return []
    class_map = {entry.value: entry.name for entry in classes}
    colors = {
        entry.value: f"#{entry.color_hint}"
        for entry in classes
        if entry.color_hint is not None
    }
    if colors:
        return [Legend(class_map=class_map, color_map=colors)]
    return [Legend(class_map=class_map)]


def _classes(legend: Legend) -> list[Classification]:
    """Spell a legend's classes as Classification classes."""
    colors = legend.color_map or {}
    classes = []
    for value, name in (legend.class_map or {}).items():
        if not _CLASS_NAME.fullmatch(name):
            raise ValueError(
                f"class name {name!r} for value {value} carries a character STAC "
                f"refuses; use only letters, digits, '-' and '_'"
            )
        hint = None
        if value in colors:
            red, green, blue = parse_color(colors[value])
            hint = f"{red:02X}{green:02X}{blue:02X}"
        classes.append(Classification.create(value=value, name=name, color_hint=hint))
    return classes
