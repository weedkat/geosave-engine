"""Ingest one model-ready raster stack on an explicit anchor."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from prefect import flow
from pydantic import JsonValue, TypeAdapter
import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io
from geosave_engine.model_spec import ModelSpec

from .anchor import AnchorConfig


def _write_stack(rasters: Mapping[str, xr.Dataset], output: str | Path) -> str:
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


@flow(name="ingest", persist_result=False)
def ingest(
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str,
) -> str:
    """Load model rasters on one anchor and write one raster stack.

    Args:
        anchor: Serializable anchor configuration defining grid and time.
        output: Destination Zarr path.
        spec: Model specification defining required rasters and STAC recipes.

    Returns:
        Path to the completed raster stack.

    Raises:
        ValueError: If a raster has no STAC recipe or fails validation.
    """
    model = ModelSpec.load(spec)
    missing = [
        name for name, requirement in model.rasters.items() if requirement.stac is None
    ]
    if missing:
        raise ValueError(f"STAC recipes required for rasters: {missing}")
    anchor_config = TypeAdapter(AnchorConfig).validate_python(anchor)
    native_anchor = anchor_config.open()

    return _write_stack(model.load_rasters(native_anchor), output)
