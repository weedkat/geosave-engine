"""Publish one completed native raster stack."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io


def write_stack(
    rasters: Mapping[str, xr.Dataset], output: str | Path, **options: Any
) -> str:
    """Write rasters fully before publishing a new local Zarr store.

    Args:
        rasters: Named, co-registered rasters.
        output: New local ``.zarr`` path.
        **options: Native Zarr writer options.

    Returns:
        Completed store path.

    Raises:
        ValueError: If the output is remote or not a ``.zarr`` path.
        FileExistsError: If the output exists before or after writing.
    """
    destination = Path(output)
    if "://" in str(output) or destination.suffix != ".zarr":
        raise ValueError("Output must be a local .zarr path")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = stack(rasters)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        io.zarr.write(tree, staged, compute=True, overwrite=False, **options)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        staged.rename(destination)
    return str(destination)
