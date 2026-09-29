"""Persist a completed native raster stack."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from pydantic import JsonValue
import xarray as xr

from geosave_engine.geodata import Dataset
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io

type SampleFormat = Literal["geotiff", "zarr"]

_PUBLICATION_OPTIONS = {"compute", "layout", "overwrite", "split_bands"}


def _write_zarr_sample(
    rasters: Mapping[str, xr.Dataset],
    output: str | Path,
    *,
    write_options: Mapping[str, JsonValue] | None = None,
) -> str:
    """Write a dense sample fully before publishing its local Zarr store."""
    destination = Path(output)
    if "://" in str(output) or destination.suffix != ".zarr":
        raise ValueError("Output must be a local .zarr path")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = stack(rasters)
    native_options = cast("dict[str, Any]", dict(write_options or {}))
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        io.zarr.write(
            tree,
            staged,
            compute=True,
            overwrite=False,
            **native_options,
        )
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        staged.rename(destination)
    return str(destination)


def open_sample(source: str | Path, *, format: SampleFormat) -> xr.DataTree:
    """Open one persisted sample as a native raster stack.

    Args:
        source: GeoTIFF sample directory or Zarr store.
        format: Persisted representation.

    Returns:
        DataTree holding the sample's named rasters.

    Raises:
        ValueError: If the format is unknown or a GeoTIFF sample has no assets.
    """
    if format == "zarr":
        return io.read_stack(source, chunks="auto")
    if format != "geotiff":
        raise ValueError(f"Unknown sample format: {format!r}")

    root = Path(source)
    assets = sorted(
        root.glob("*.tif"), key=lambda path: (path.stem != "label", path.name)
    )
    if not assets:
        raise ValueError(f"GeoTIFF sample has no .tif assets: {root}")
    return stack(
        {asset.stem: io.read_raster(asset, chunks="auto") for asset in assets}
    )


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
        output: GeoTIFF sample directory or Zarr store.
        format: Persisted representation.
        write_options: Serializable encoding options for the native writer.

    Returns:
        Completed sample path.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If the format, options, or GeoTIFF time axis is invalid.
    """
    options = dict(write_options or {})
    if reserved := sorted(options.keys() & _PUBLICATION_OPTIONS):
        raise ValueError(f"GeoSave owns sample publication options: {reserved}")
    if format == "zarr":
        return _write_zarr_sample(rasters, output, write_options=options)
    if format != "geotiff":
        raise ValueError(f"Unknown sample format: {format!r}")

    destination = Path(output)
    if "://" in str(output):
        raise ValueError("Output must be a local sample directory")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    scenes = {name: _geotiff_scene(raster, name) for name, raster in rasters.items()}
    native_options = cast("dict[str, Any]", options)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.mkdir()
        for name, scene in scenes.items():
            io.geotiff.write_cog(
                cast("Dataset", scene),
                staged / f"{name}.tif",
                **native_options,
            )
        with open_sample(staged, format="geotiff") as restored:
            if set(restored.gs.groups) != set(scenes):
                raise ValueError("Completed GeoTIFF assets do not match sample rasters")
            _ = restored.gs.anchor
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        staged.rename(destination)
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
