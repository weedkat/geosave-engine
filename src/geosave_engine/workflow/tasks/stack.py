"""Publish one completed native raster stack."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io


def write_stack(rasters: Mapping[str, xr.Dataset], output: str | Path) -> str:
    """Write rasters fully before publishing a new local Zarr store."""
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
        io.zarr.write(tree, staged, compute=True, overwrite=False)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        staged.rename(destination)
    return str(destination)
