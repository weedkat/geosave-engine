"""Execute one model-owned call as an observable Prefect task."""

from collections.abc import Mapping

from prefect import task
from prefect.cache_policies import NO_CACHE

from geosave_engine.workflow.specs import CallSpec


@task(name="model-call", cache_policy=NO_CACHE, persist_result=False)
def invoke_call(
    spec: CallSpec,
    inputs: Mapping[str, object],
    /,
) -> object:
    """Invoke one model-owned call as an observable task run."""
    return spec.invoke(inputs)
