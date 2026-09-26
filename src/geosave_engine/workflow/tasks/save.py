"""Persist a completed native raster stack."""

from pathlib import Path
from tempfile import TemporaryDirectory

from prefect import task
from prefect.cache_policies import NO_CACHE
import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io


def write_stack(rasters: dict[str, xr.Dataset], output: str | Path) -> str:
    """Write named rasters fully before publishing a new local Zarr store."""
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


@task(cache_policy=NO_CACHE, persist_result=False)
def save_stack(rasters: dict[str, xr.Dataset], output: str | Path) -> str:
    """Write a raster stack through the transitional Prefect task."""
    return write_stack(rasters, output)
