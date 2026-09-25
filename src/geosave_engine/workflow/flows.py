"""Prefect orchestration for native STAC acquisition."""

from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner
from pydantic import JsonValue

from geosave_engine.geodata.core.stack import stack

from .ingestion import acquire
from .io import _destination, write_stack
from .runtime import open_anchor, open_sources
from .spec import ModelSpec


# Durable checkpoints are explicit writes, not serialized native lazy graphs.
_acquire = task(acquire, cache_policy=None, persist_result=False)
_write_stack = task(write_stack, cache_policy=None, persist_result=False)


@flow(
    name="ingest",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    persist_result=False,
)
def ingest(
    sources: dict[str, dict[str, JsonValue]],
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str,
) -> str:
    """Acquire STAC rasters from primitive settings into a completed local stack.

    Args:
        sources: Model source names mapped to optional query and load settings.
        anchor: Explicit coordinates or GeoJSON path with native grid settings.
        output: New local .zarr store.
        spec: Model YAML or artifact directory owning collections and endpoints.

    Returns:
        Path of the completed raw raster stack.

    Raises:
        ValueError: Source, anchor or output settings are invalid.
        FileExistsError: The destination already exists.
    """
    # Check the output and select the sources required by the model.
    destination = _destination(output)
    requirements = ModelSpec.load(spec).sources
    if not sources:
        raise ValueError("At least one source is required")
    if missing := requirements.keys() - sources.keys():
        raise ValueError(f"Source bindings are missing: {sorted(missing)}")
    if extra := sources.keys() - requirements.keys():
        raise ValueError(f"Unknown source bindings: {sorted(extra)}")
    selected = {name: sources[name] for name in requirements}
    # Acquire each source as a separate monitored job.
    region = open_anchor(anchor)
    bound = open_sources(selected, requirements)
    pending = {
        name: _acquire.submit(
            {name: source},
            region,
            requirements={name: requirements[name]},
        )
        for name, source in bound.items()
    }
    # Combine the rasters and return a completed file path.
    raw = stack(
        {name: result.result().gs.rasters[name] for name, result in pending.items()}
    )
    return _write_stack(raw, destination)
