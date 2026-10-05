"""How a cube is arranged as a tree of raster files.

A rasterio leaf holds one `(band, y, x)` array, where GDAL's `band` means a
spectral band. So a leaf's bands are data variables, every other axis reaches
the path, and a layout arranges only those path parts.

Examples:
    A cube over two instants with variables `B04` and `B08`, written with
    `split_bands=True`::

        layout="nested"                layout="flat"
        scene/                         scene/
          20250601T103031/               20250601T103031_B04.tif
            B04.tif                      20250601T103031_B08.tif
            B08.tif                      20250611T103031_B04.tif
          20250611T103031/               20250611T103031_B08.tif
            B04.tif
            B08.tif

    With `split_bands=False` the variables stay bands of one file per
    instant, so both arrangements give `scene/20250601T103031.tif`.

Instants are spelled by `utils.datetime.format_instant`, matching the granule
stamps ESA writes. Time is never a leaf's bands: band descriptions naming
timestamps read as nothing to a GIS.

Leaves are COGs. Every other GDAL driver either carries no band descriptions,
which is how variable names survive, or holds no georeference of its own — so
a tree written with one could not be read back.

Only writing varies. A leaf names its own variables and instant, so `read_tree`
rebuilds a cube from the files rather than from their paths, and every
arrangement reads the same way. A vendor tree such as ESA's SAFE is therefore
not an arrangement but a different reader: its leaves are JP2 and its metadata
sits in sidecar XML, so none of this applies to it.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import pandas as pd
import xarray as xr

from geosave_engine.geodata.attrs import merge, rebase
from geosave_engine.geodata.core.profile import TIME_COORDINATE

from geosave_engine.geodata.utils.datetime import format_instant

from . import gdal
from .geotiff import write_cog

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset

_SUFFIX: Final = ".tif"


# Where one leaf sits relative to the root, given its instant and its variable.
type LeafPath = Callable[[str, str], Path]


def nested(timestamp: str, variable: str) -> Path:
    """Put each instant on its own directory level, variables inside it.

    One instant is one directory holding its bands, which is how ESA, USGS,
    and STAC-backed COG collections all ship a scene.

    Args:
        timestamp: Instant token, empty for a cube carrying no time axis.
        variable: Data variable name.

    Returns:
        Leaf path relative to the tree's root, without a suffix.

    Examples:
        >>> nested("20250601T103031", "B04")
        PosixPath('20250601T103031/B04')
    """
    return Path(timestamp) / variable if timestamp else Path(variable)


def flat(timestamp: str, variable: str, sep: str = "_") -> Path:
    """Join each instant to its variable inside one filename, in one directory.

    For an object store, where prefixes are cheap and directories are not
    real, and for handing someone a single folder.

    Args:
        timestamp: Instant token, empty for a cube carrying no time axis.
        variable: Data variable name.
        sep: Joins the instant to the variable.

    Returns:
        Leaf path relative to the tree's root, without a suffix.

    Examples:
        >>> flat("20250601T103031", "B04")
        PosixPath('20250601T103031_B04')
    """
    return Path(f"{timestamp}{sep}{variable}" if timestamp else variable)


LAYOUTS: dict[str, LeafPath] = {"nested": nested, "flat": flat}


def write_tree(
    ds: Dataset,
    destination: str | PathLike[str],
    *,
    layout: str | LeafPath = "nested",
    split_bands: bool = False,
    map_scale: float | None = None,
    overwrite: bool = False,
    **rio_options: Any,
) -> Path:
    """Write a cube as a tree of COG leaves.

    Args:
        ds: Cube to write.
        destination: Directory the tree is written into, or the file path when
            the cube carries no time axis and `split_bands` is off.
        layout: A name in `LAYOUTS`, or a callable placing one leaf from its
            instant and variable. Ignored unless `split_bands` is set, since
            a whole scene's leaf is named by its instant alone.
        split_bands: True gives each variable its own single-band file. False
            keeps every variable as a band of one file, which needs them to
            share a dtype.
        map_scale: Map denominator used to write pixels per centimetre in
            every leaf.
        overwrite: Replace leaves that already exist.
        **rio_options: COG creation options passed to every leaf.

    Returns:
        The directory written, or the one file a timeless cube with
        `split_bands` off becomes. `read_raster` opens either.

    Raises:
        FileExistsError: A leaf exists and `overwrite` is false.
        KeyError: `layout` names no known arrangement.
        ValueError: `ds` carries no locatable grid, or spans a non-spatial
            axis other than time.

    Examples:
        >>> write_tree(cube, "scene")  # scene/20250601T103031/B04.tif, ...
        >>> write_tree(cube, "scene", layout="flat", split_bands=True)
    """
    place = LAYOUTS[layout] if isinstance(layout, str) else layout
    root = Path(destination)
    for timestamp, scene in scenes(ds):
        if not split_bands:
            # An instant names its own leaf, so only a timeless one is the destination.
            leaf = (root / timestamp) if timestamp else root
            written = write_cog(
                scene,
                leaf.with_suffix(_SUFFIX),
                map_scale=map_scale,
                overwrite=overwrite,
                **rio_options,
            )
            if not timestamp:
                return written
            continue
        for name in scene.data_vars:
            leaf = root / place(timestamp, str(name))
            write_cog(
                scene[[name]],
                leaf.with_suffix(_SUFFIX),
                map_scale=map_scale,
                overwrite=overwrite,
                **rio_options,
            )
    return root


def scenes(ds: Dataset) -> list[tuple[str, Dataset]]:
    """Cut a cube into the one-instant scenes its leaves are written from.

    A leaf holds one instant of one grid, so a cube spanning any other
    non-spatial axis is refused rather than flattened.

    Args:
        ds: Cube about to be written.

    Returns:
        Each instant's filename token paired with the scene at it. A cube
        carrying no time axis yields one pair whose token is empty.

    Raises:
        ValueError: `ds` carries no locatable grid, spans a non-spatial axis
            other than time, or the time axis holds NaT or sub-second
            precision.

    Examples:
        >>> scenes(cube)
        [('20250601T103031', <xarray.Dataset> Size: 2MB ...),
         ('20250611T103031', <xarray.Dataset> Size: 2MB ...)]
        >>> scenes(dem)
        [('', <xarray.Dataset> Size: 1MB ...)]
    """
    spatial = ds.odc.spatial_dims
    if spatial is None:
        raise ValueError(
            "the cube carries no locatable grid, so its leaves cannot be "
            "georeferenced; assign a CRS with odc.geo.xr.assign_crs first"
        )
    # Only pixels have to fit a leaf; an axis only a coordinate spans writes nothing.
    pixel_dims = {str(dim) for array in ds.data_vars.values() for dim in array.dims}
    beyond = sorted(pixel_dims - {*spatial, TIME_COORDINATE})
    if beyond:
        raise ValueError(
            f"a leaf holds one instant of one grid, but the cube also spans "
            f"{beyond}; select those axes away before writing"
        )
    if TIME_COORDINATE not in ds.dims:
        return [("", ds)]

    cut: list[tuple[str, Dataset]] = []
    for value in ds[TIME_COORDINATE].values:
        instant = pd.Timestamp(value).to_pydatetime()
        if not isinstance(instant, datetime):  # NaT
            raise ValueError(
                "the time axis holds NaT, so no leaf can be named for it; drop "
                "the empty step before writing"
            )
        cut.append((format_instant(instant), ds.sel({TIME_COORDINATE: value})))
    return cut


def read_tree(source: str | PathLike[str], **rio_options: Any) -> Dataset:
    """Build a cube from every COG leaf under a path.

    A leaf names its own variables in its band descriptions and its own instant
    in `TIFFTAG_DATETIME`, so the cube is rebuilt from the files. Nested and
    flat trees therefore read the same way.

    Args:
        source: Directory holding a tree, or one leaf on its own.
        **rio_options: Open options passed to every leaf.

    Returns:
        The cube the leaves hold, its time axis in ascending order.

    Raises:
        ValueError: `source` holds no leaf, its leaves name no variables, or a
            model refuses what the leaves disagree on.

    Warns:
        DroppedAttrsWarning: The leaves carry an attr differently and its model
            drops rather than refuses the disagreement.

    Examples:
        >>> read_tree("scene").gs.variables
        ('B04', 'B08')
    """
    root = Path(source)
    if root.is_dir():
        paths = sorted(root.rglob(f"*{_SUFFIX}"))
    else:
        paths = [root.with_suffix(_SUFFIX)]
    if not paths:
        raise ValueError(f"{root} holds no {_SUFFIX} leaf to read")

    with ExitStack() as opened:
        leaves: list[xr.Dataset] = []
        for path in paths:
            leaf = gdal.read(path, **rio_options)
            opened.callback(leaf.close)
            if TIME_COORDINATE in leaf.coords and TIME_COORDINATE not in leaf.dims:
                leaf = leaf.expand_dims(TIME_COORDINATE)
            leaves.append(leaf)

        # Attrs drop here and are rebased below, where each model rules on its own.
        cube = xr.combine_by_coords(
            leaves, compat="no_conflicts", join="outer", combine_attrs="drop"
        )
        if not isinstance(cube, xr.Dataset):
            raise ValueError(
                f"{root} holds leaves naming no variables, so they combine into an "
                f"array rather than a cube; write them with band descriptions"
            )
        cube = rebase(cube, merge(leaves))
        # The tree, not any one leaf, is what a caller reopens.
        cube.encoding["source"] = str((root if root.is_dir() else paths[0]).resolve())
        cube.set_close(opened.pop_all().close)
        return cast("Dataset", cube)
