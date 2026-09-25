"""Orchestrate model-owned preprocessing declarations."""

from collections.abc import Mapping

from prefect import flow

from geosave_engine.workflow.specs import ModelSpec
from geosave_engine.workflow.tasks import invoke_call


@flow(name="preprocess", persist_result=False)
def preprocess(
    inputs: Mapping[str, object],
    spec: ModelSpec,
) -> dict[str, object]:
    """Run each declared preprocessing call as a named task run."""
    model = spec.validated_copy()
    stage = model.preprocessing
    required = stage.validate_inputs(inputs.keys())
    state = dict(inputs)

    for name in stage.external_inputs:
        if name not in required or name not in model.sources:
            continue
        try:
            state[name] = model.sources[name].select_raster(state[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Source {name!r}: {error}") from error

    futures = {}
    for name, declaration in stage.items():
        operation = invoke_call.with_options(task_run_name=f"preprocess-{name}")
        future = operation.submit(declaration, declaration.select_inputs(state))
        state[name] = future
        futures[name] = future

    return {name: future.result() for name, future in futures.items()}
