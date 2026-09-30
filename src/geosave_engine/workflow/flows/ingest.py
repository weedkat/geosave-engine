"""Ingest one model-ready raster stack on an explicit anchor."""

from prefect import flow
from pydantic import JsonValue, TypeAdapter

from geosave_engine.model_spec import ModelSpec
from geosave_engine.workflow.configs import AnchorConfig
from geosave_engine.workflow.tasks.stack import write_stack



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
    model.require_recipes()
    anchor_config = TypeAdapter(AnchorConfig).validate_python(anchor)
    native_anchor = anchor_config.open()

    return write_stack(model.load_rasters(native_anchor), output)
