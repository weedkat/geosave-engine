"""Ingest one model-ready raster stack on an explicit anchor."""

from prefect import flow
from pydantic import JsonValue, TypeAdapter

from geosave_engine.workflow.configs import AnchorConfig, SourceConfig
from geosave_engine.workflow.specs import ModelSpec
from geosave_engine.workflow.tasks.load import RasterLoader
from geosave_engine.workflow.tasks.save import write_stack


@flow(name="ingest", persist_result=False)
def ingest(
    sources: dict[str, dict[str, JsonValue]],
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str,
) -> str:
    """Load model sources on one anchor and write one raster stack."""
    configs = {
        name: SourceConfig.model_validate(config) for name, config in sources.items()
    }
    native_anchor = TypeAdapter(AnchorConfig).validate_python(anchor).open()
    requirements = ModelSpec.load(spec).sources
    if configs.keys() != requirements.keys():
        raise ValueError("Source bindings must match model sources")

    rasters = {
        name: RasterLoader(requirement).load(configs[name], native_anchor)
        for name, requirement in requirements.items()
    }
    return write_stack(rasters, output)
