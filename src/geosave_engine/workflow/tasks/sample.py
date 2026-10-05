"""Publish one sample: a folder holding one raster per layer."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from pydantic import JsonValue
import xarray as xr

from geosave_engine.geodata import Dataset
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata import io

type SampleFormat = Literal["geotiff", "zarr"]

_SUFFIXES = {"geotiff": ".tif", "zarr": ".zarr"}
_PUBLICATION_OPTIONS = {"compute", "layout", "overwrite", "split_bands"}


def write_sample(
    rasters: Mapping[str, xr.Dataset],
    output: str | Path,
    *,
    format: SampleFormat = "geotiff",
    write_options: Mapping[str, JsonValue] | None = None,
) -> str:
    """Atomically publish one named dense-training sample.

    Args:
        rasters: Named, co-registered sample rasters.
        output: Sample directory, which takes one raster per name.
        format: Persisted representation of each raster. GeoTIFF holds one
            instant; Zarr holds a time series.
        write_options: Serializable encoding options for the native writer.

    Returns:
        Completed sample path.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If the options or GeoTIFF time axis is invalid, the
            rasters do not share a grid, or the output names a store.
    """
    options = cast("dict[str, Any]", dict(write_options or {}))
    if reserved := sorted(options.keys() & _PUBLICATION_OPTIONS):
        raise ValueError(f"GeoSave owns sample publication options: {reserved}")

    destination = Path(output)
    if "://" in str(output):
        raise ValueError("Output must be a local sample directory")
    if destination.suffix.lower() in (".zarr", ".safe"):
        raise ValueError(
            f"Output {destination} names a store; a sample is a plain directory "
            f"holding one raster per layer"
        )
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if stack(rasters).gs.geobox is None:
        raise ValueError(f"Sample rasters do not share a grid: {list(rasters)}")

    if format == "geotiff":
        rasters = {
            name: _geotiff_scene(raster, name) for name, raster in rasters.items()
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as staging:
        draft = Path(staging) / destination.name
        draft.mkdir()
        for name, raster in rasters.items():
            path = draft / f"{name}{_SUFFIXES[format]}"
            if format == "geotiff":
                io.geotiff.write_cog(cast("Dataset", raster), path, **options)
            else:
                io.zarr.write(raster, path, **options)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        draft.rename(destination)
    return str(destination)


def _geotiff_scene(raster: xr.Dataset, name: str) -> xr.Dataset:
    """Reduce one singleton time axis for flat GeoTIFF persistence."""
    if "time" not in raster.dims:
        return raster
    if raster.sizes["time"] != 1:
        raise ValueError(
            f"GeoTIFF raster {name!r} has {raster.sizes['time']} time steps; "
            "use format='zarr'"
        )
    return raster.squeeze("time", drop=False)
