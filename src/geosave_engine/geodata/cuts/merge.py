"""Merge chip outputs saved during prediction back into whole rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from tiler import Merger

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS
from geosave_engine.geodata.core.raster import raster

from .chips import layout

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset


def merge(
    parents: gpd.GeoDataFrame,
    windows: gpd.GeoDataFrame,
    outputs: xr.DataArray,
    *,
    taper: str | None = None,
) -> dict[str, Dataset]:
    """Merge the outputs of chips into one raster per window they were cut from.

    The layout a cut used is rebuilt from the same shape and settings, so
    nothing but the two tables and the outputs is needed, and chips may
    arrive in any order and from several writers.

    Args:
        parents: The windows the chips were cut from, as given to `chips`.
        windows: The chips, as `chips` returned them.
        outputs: One output per chip along `id`, shaped `(id, y, x)` or with
            one more axis between, as `(id, band, y, x)`.
        taper: How overlapping chips are weighed, as a window name such as
            `"hann"`. None weighs every chip alike.

    Returns:
        {
            "<parent id>": Dataset on the parent's grid, holding the merged
                output under the name `outputs` carries, or `"output"`,
        }
        One entry per parent that has chips in `windows`.

    Raises:
        ValueError: A parent's chip has no output.

    Examples:
        >>> merged = merge(frames, chips, outputs, taper="hann")
        >>> merged["s0/frame-0"]["logits"].shape
        (3, 512, 512)
    """
    name = str(outputs.name) if outputs.name is not None else "output"
    leading = [dim for dim in outputs.dims if dim not in ("id", *SPATIAL_DIMENSIONS)]
    available = set(outputs["id"].values.tolist())

    merged: dict[str, Dataset] = {}
    for parent in parents.to_dict("records"):
        chips = windows[windows["parent"] == parent["id"]]
        if chips.empty:
            continue
        absent = [chip for chip in chips["id"] if chip not in available]
        if absent:
            raise ValueError(
                f"{parent['id']!r} cannot be merged: its chips {absent} have no "
                f"output; predict them first"
            )

        first = chips.iloc[0]
        shape = (int(parent["height"]), int(parent["width"]))
        tiler, halo = layout(
            shape,
            (int(first["height"]), int(first["width"])),
            overlap=int(first["overlap"]),
            mode=first["mode"],
        )
        merger = Merger(
            tiler,
            logits=outputs.sizes[leading[0]] if leading else 0,
            window=taper,
            save_visits=False,
        )
        for chip in chips.itertuples():
            merger.add(int(chip.chip), outputs.sel(id=chip.id).values)
        values = merger.merge(extra_padding=halo)

        dims = (*leading, *SPATIAL_DIMENSIONS)
        # Unreferenced pixels merge as pixels; a located parent keeps its grid.
        if isinstance(parent["crs"], str):
            grid = GeoBox(shape, Affine(*parent["transform"]), parent["crs"])
            merged[parent["id"]] = raster({name: (dims, values)}, grid)
        else:
            merged[parent["id"]] = cast("Dataset", xr.Dataset({name: (dims, values)}))
    return merged
