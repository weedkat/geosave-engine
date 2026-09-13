"""How a cube is arranged as a tree of raster files.

A rasterio leaf holds one `(band, y, x)` array, where GDAL's `band` means a
spectral band. So a leaf's bands are data variables, every other axis reaches
the path, and a layout arranges only those path parts.

Examples:
    A cube over two instants with variables `B04` and `B08` writes as::

        NestedLayout()                 FlatLayout()
        scene/                         scene/
          20250601T103031/               20250601T103031_B04.tif
            B04.tif                      20250601T103031_B08.tif
            B08.tif                      20250611T103031_B04.tif
          20250611T103031/               20250611T103031_B08.tif
            B04.tif
            B08.tif

Instants are spelled by `utils.datetime.format_instant`, matching the granule
stamps ESA writes. Time is never a leaf's bands: band descriptions naming
timestamps read as nothing to a GIS.

Leaves are COGs. Every other GDAL driver either carries no band descriptions,
which is how variable names survive, or holds no georeference of its own — so
a tree written with one could not be read back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast, runtime_checkable

import pandas as pd
import xarray as xr

from geosave_engine.geodata.attrs import merge, rebase
from geosave_engine.geodata.core.convention import TIME_COORDINATE

from ..datetime import format_instant

from .gdal import read as read_raster
from .geotiff import write_cog

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset

_SUFFIX: Final = ".tif"


@runtime_checkable
class Layout(Protocol):
    """Write a cube as a tree of raster files, and build it back.

    `read` and `write` are inverses: reading what a layout wrote rebuilds the
    Dataset, coordinates and variable names included.
    """

    def write(
        self,
        ds: Dataset,
        destination: str | PathLike[str],
        *,
        overwrite: bool = False,
        **rio_options: Any,
    ) -> None:
        """Write `ds` as a tree of COG leaves under `destination`.

        Args:
            ds: Cube to write.
            destination: Directory the tree is written into.
            overwrite: Replace leaves that already exist.
            **rio_options: COG creation options passed to every leaf.

        Raises:
            NotImplementedError: The layout describes someone else's tree.
            ValueError: `ds` carries no locatable grid, or spans a non-spatial
                axis other than time.
        """
        ...

    def read(
        self,
        source: str | PathLike[str],
        **rio_options: Any,
    ) -> Dataset:
        """Build a cube from the tree under `source`.

        Args:
            source: Directory holding the tree.
            **rio_options: Open options passed to every leaf.

        Returns:
            The cube the tree holds.

        Raises:
            ValueError: The tree is ragged against this layout, or its leaves
                disagree on grid, dtype, or fill value.
        """
        ...


@dataclass(frozen=True)
class NestedLayout:
    """One directory level per axis, instant outermost and variable innermost.

    One instant is one directory holding its bands, which is how ESA, USGS,
    and STAC-backed COG collections all ship a scene.

    Args:
        split_bands: True gives each variable its own single-band file. False
            keeps every variable as a band of one file, which needs them to
            share a dtype.

    Examples:
        `split_bands=True`, two instants, variables `B04` and `B08`::

            scene/
              20250601T103031/B04.tif
              20250601T103031/B08.tif
              20250611T103031/B04.tif
              20250611T103031/B08.tif

        `split_bands=False` drops the variable level, so the file is the
        instant::

            scene/
              20250601T103031.tif   bands: B04, B08
              20250611T103031.tif   bands: B04, B08

        A cube with no time axis is one file, named by the destination::

            dem.tif   bands: elevation, slope
    """

    split_bands: bool = True

    def write(
        self,
        ds: Dataset,
        destination: str | PathLike[str],
        *,
        overwrite: bool = False,
        **rio_options: Any,
    ) -> None:
        """Write one directory per instant, its bands inside."""
        root = Path(destination)
        for timestamp, scene in scenes(ds):
            if not self.split_bands:
                one_file = (root / timestamp) if timestamp else root
                write_cog(
                    scene,
                    one_file.with_suffix(_SUFFIX),
                    overwrite=overwrite,
                    **rio_options,
                )
                continue
            for name in scene.data_vars:
                write_cog(
                    scene[[name]],
                    root / timestamp / f"{name}{_SUFFIX}",
                    overwrite=overwrite,
                    **rio_options,
                )

    def read(
        self,
        source: str | PathLike[str],
        **rio_options: Any,
    ) -> Dataset:
        """Read each instant directory as one step of the time axis."""
        return read_tree(source, **rio_options)


@dataclass(frozen=True)
class FlatLayout:
    """One directory, axis values joined into each filename.

    For an object store, where prefixes are cheap and directories are not
    real, and for handing someone a single folder.

    Args:
        sep: Joins axis values within a filename.
        split_bands: True gives each variable its own single-band file. False
            keeps every variable as a band of one file, which needs them to
            share a dtype.

    Examples:
        `split_bands=True`, two instants, variables `B04` and `B08`::

            scene/
              20250601T103031_B04.tif
              20250601T103031_B08.tif
              20250611T103031_B04.tif
              20250611T103031_B08.tif

        `split_bands=False` leaves only the instant in each name::

            scene/
              20250601T103031.tif   bands: B04, B08
              20250611T103031.tif   bands: B04, B08

        A cube with no time axis is one file, named by the destination::

            dem.tif   bands: elevation, slope
    """

    sep: str = "_"
    split_bands: bool = True

    def write(
        self,
        ds: Dataset,
        destination: str | PathLike[str],
        *,
        overwrite: bool = False,
        **rio_options: Any,
    ) -> None:
        """Write one directory, each leaf named by its axis values."""
        root = Path(destination)
        for timestamp, scene in scenes(ds):
            if not self.split_bands:
                one_file = (root / timestamp) if timestamp else root
                write_cog(
                    scene,
                    one_file.with_suffix(_SUFFIX),
                    overwrite=overwrite,
                    **rio_options,
                )
                continue
            for name in scene.data_vars:
                stem = f"{timestamp}{self.sep}{name}" if timestamp else str(name)
                write_cog(
                    scene[[name]],
                    root / f"{stem}{_SUFFIX}",
                    overwrite=overwrite,
                    **rio_options,
                )

    def read(
        self,
        source: str | PathLike[str],
        **rio_options: Any,
    ) -> Dataset:
        """Read each filename, split on `sep`, as one leaf's axis values."""
        return read_tree(source, **rio_options)


@dataclass(frozen=True)
class SAFELayout:
    """ESA's SAFE product tree, whose shape ESA fixes.

    Not an arrangement this library chooses, so it takes no arguments. The
    resolution groups sit on different grids, so one product does not read
    back as one cube; which group to read is a question for the reader.

    Examples:
        A Sentinel-2 Level-2A product::

            S2A_MSIL2A_20250601T103031_N0511_R108_T33UUP_....SAFE/
              MTD_MSIL2A.xml                 product metadata
              manifest.safe                  package manifest and checksums
              GRANULE/L2A_T33UUP_A051234_20250601T103031/
                MTD_TL.xml                   tile metadata, angle grids
                QI_DATA/                     cloud and quality masks
                IMG_DATA/
                  R10m/T33UUP_20250601T103031_B04_10m.jp2
                  R20m/T33UUP_20250601T103031_B05_20m.jp2
                  R60m/T33UUP_20250601T103031_B01_60m.jp2

        A leaf is named `T{tile}_{instant}_{band}_{resolution}m.jp2`, so the
        band and the instant are read from the filename rather than from
        metadata inside the file.
    """

    def write(
        self,
        ds: Dataset,
        destination: str | PathLike[str],
        *,
        overwrite: bool = False,
        **rio_options: Any,
    ) -> None:
        """Refuse: ESA owns a product's identifiers and checksums.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "a SAFE product carries ESA's own identifiers and checksums, so "
            "writing one would claim a provenance this library cannot honour"
        )

    def read(
        self,
        source: str | PathLike[str],
        **rio_options: Any,
    ) -> Dataset:
        """Read one granule's JP2 leaves, their band and instant from each name.

        Raises:
            NotImplementedError: Always; reading a SAFE granule is unbuilt.
        """
        raise NotImplementedError(
            "reading a SAFE granule is not built yet; open its JP2 leaves with "
            "gdal.read and stack them, or use a flat or nested layout"
        )


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

    leaves: list[xr.Dataset] = []
    for path in paths:
        leaf = read_raster(path, **rio_options)
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
    return cast("Dataset", cube)
