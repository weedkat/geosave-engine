"""How a raster is arranged as Cloud Optimized GeoTIFF files.

A COG holds one `(band, y, x)` array: one instant of one grid, its bands the
data variables. A raster with a time dimension is therefore one scene per file
or folder, each named after the path the raster was written to.

Examples:
    `write(ds, "samples/forest")` for a raster over two instants::

        split_bands=False                      split_bands=True
        forest/                                forest/
          forest_20250601T103031.tif             forest_20250601T103031/
          forest_20250611T103031.tif               B04.tif
                                                   B08.tif
                                                 forest_20250611T103031/
                                                   B04.tif
                                                   B08.tif

    Without a time dimension the same calls give `forest.tif`, or
    `forest/B04.tif` and `forest/B08.tif`.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NamedTuple, Unpack, cast

import pandas as pd
import xarray as xr

from geosave_engine.geodata.conventions import TIME_COORDINATE
from geosave_engine.geodata.utils.datetime import format_instant

from .geotiff import COGWriteOptions, write_cog
from ..storage import StorageOptions

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset


class CogFile(NamedTuple):
    """One COG a raster is written as.

    Args:
        scene: Name of the scene the file belongs to: the raster's name, with
            its instant where the raster has a time dimension.
        path: Where the file goes, as text.
        raster: Pixels the file holds, without a time dimension.
        time: Instant the file holds, or None where the raster has no time
            dimension.
    """

    scene: str
    path: str
    raster: xr.Dataset
    time: pd.Timestamp | None


def layout(
    ds: xr.Dataset, path: str | PathLike[str], *, split_bands: bool = False
) -> list[CogFile]:
    """Say which COG files a raster is written as, without writing them.

    Args:
        ds: Raster to arrange.
        path: Name of the raster, as a local path or fsspec URL.
        split_bands: True gives each variable its own single-band file. A
            raster with one variable is one file per scene either way.

    Returns:
        One entry per file, in time then variable order.

    Raises:
        ValueError: `ds` carries no locatable grid, spans a non-spatial axis
            other than time, or its instants are empty, repeated or NaT.

    Examples:
        >>> layout(ds, "samples/forest")[0].path
        'samples/forest/forest_20250601T103031.tif'
    """
    spatial = ds.odc.spatial_dims
    if spatial is None:
        raise ValueError(
            "the raster carries no locatable grid, so its files cannot be "
            "georeferenced; assign a CRS with odc.geo.xr.assign_crs first"
        )
    # Only pixels have to fit a file; an axis only a coordinate spans writes nothing.
    pixel_dims = {str(dim) for array in ds.data_vars.values() for dim in array.dims}
    extra_dims = sorted(pixel_dims - {*spatial, TIME_COORDINATE})
    if extra_dims:
        raise ValueError(
            f"a COG holds one instant of one grid, but the raster also spans "
            f"{extra_dims}; select those axes away before writing"
        )

    # Names are joined as text, so a URL keeps its `scheme://`.
    root = str(path).rstrip("/")
    name = PurePosixPath(root).name

    # Each instant is one scene, named after the raster and its time. A raster
    # without a time dimension is one scene named after the raster alone.
    scenes: list[tuple[str, str, xr.Dataset, pd.Timestamp | None]] = []
    if TIME_COORDINATE in ds.dims:
        times = pd.DatetimeIndex(ds[TIME_COORDINATE].values)
        if times.hasnans or not times.is_unique or len(times) == 0:
            raise ValueError("scene times must be nonempty, unique and not NaT")
        for index, instant in enumerate(times):
            scene = f"{name}_{format_instant(instant)}"
            pixels = ds.isel({TIME_COORDINATE: index})
            scenes.append((scene, f"{root}/{scene}", pixels, instant))
    else:
        scenes.append((name, root, ds, None))

    # One variable is already one band per file, and its file keeps the scene's name.
    split = split_bands and len(ds.data_vars) > 1

    # A split scene is a folder holding one file per band; any other is one file.
    files: list[CogFile] = []
    for scene, scene_path, pixels, instant in scenes:
        if split:
            for band in pixels.data_vars:
                file = CogFile(
                    scene, f"{scene_path}/{band}.tif", pixels[[band]], instant
                )
                files.append(file)
        else:
            file = CogFile(scene, f"{scene_path}.tif", pixels, instant)
            files.append(file)
    return files


def write(
    ds: xr.Dataset,
    path: str | PathLike[str],
    *,
    split_bands: bool = False,
    map_scale: float | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> tuple[Path | str, ...]:
    """Write a raster as COG files named after `path`.

    Args:
        ds: Raster to write.
        path: Name of the raster, as a local path or fsspec URL. It is never
            read as a filename: `.tif` is appended to it or to the files in it.
        split_bands: True gives each variable its own single-band file. A
            raster with one variable is written as one file either way.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace files that already exist.
        storage_options: Options for the filesystem a URL names.
        **options: COG creation options passed to every file.

    Returns:
        Exactly the files written, in time then variable order.

    Raises:
        FileExistsError: A file exists and `overwrite` is false.
        ValueError: `ds` carries no locatable grid, spans a non-spatial axis
            other than time, or its instants are empty, repeated or NaT.

    Examples:
        >>> write(ds, "samples/forest")[0]
        PosixPath('samples/forest/forest_20250601T103031.tif')
    """
    written = []
    for file in layout(ds, path, split_bands=split_bands):
        written_path = write_cog(
            cast("Dataset", file.raster),
            file.path,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )
        written.append(written_path)
    return tuple(written)
