"""Ingest one model-ready raster sample on an explicit anchor."""

from typing import Literal

from prefect import flow
from pydantic import JsonValue

from geosave_engine.model.spec import ModelSpec
from geosave_engine.workflow.configs import AnchorConfig
from geosave_engine.workflow.tasks.sample import write_sample


@flow(name="ingest", persist_result=False)
def ingest(
    anchor: AnchorConfig,
    *,
    output: str,
    spec: str,
    format: Literal["geotiff", "zarr"] = "zarr",
    write_options: dict[str, JsonValue] | None = None,
) -> str:
    """Load model rasters on one anchor and write them as one sample.

    Args:
        anchor: Anchor configuration defining grid and time.
        output: New directory, which takes one raster per model raster.
        spec: Model specification defining required rasters and STAC recipes.
        format: Persisted representation of each raster. GeoTIFF holds one
            instant; Zarr holds a time series.
        write_options: Serializable options for the native raster writer.

    Returns:
        Path to the completed sample directory.

    Raises:
        FileExistsError: If the output directory already exists.
        ValueError: If a raster declares no `stac` block or fails validation.
    """
    rasters = ModelSpec.load(spec).load_rasters(anchor.open())
    return write_sample(rasters, output, format=format, write_options=write_options)
