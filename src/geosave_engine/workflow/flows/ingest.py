"""Deployable ingestion flow from primitive parameters to a completed stack."""

from prefect import flow
from pydantic import JsonValue

from geosave_engine.workflow.configs import IngestConfig
from geosave_engine.workflow.specs import ModelSpec
from geosave_engine.workflow.tasks import load_raster, save_stack


@flow(
    name="ingest",
    persist_result=False,
)
def ingest(
    sources: dict[str, dict[str, JsonValue]],
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str,
) -> str:
    """Load required STAC rasters and publish one completed local Zarr stack."""
    config = IngestConfig.model_validate(
        {"sources": sources, "anchor": anchor, "output": output, "spec": spec}
    )
    model = ModelSpec.load(config.spec)
    config.validate_sources(model.sources)
    pending = {
        name: load_raster.submit(name, config.sources[name], config.anchor, requirement)
        for name, requirement in model.sources.items()
    }
    rasters = {name: result.result() for name, result in pending.items()}
    return save_stack.submit(rasters, config.output).result()
