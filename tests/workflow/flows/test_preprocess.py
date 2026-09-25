from __future__ import annotations

from threading import Barrier

from dask.callbacks import Callback
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import TaskRunFilter, TaskRunFilterName
import pytest
import xarray as xr

from geosave_engine.workflow.flows import preprocess
from geosave_engine.workflow.specs import ModelSpec, Ref
from geosave_engine.workflow.tasks import invoke_call

_barrier: Barrier | None = None


def meet(*, value: int) -> int:
    if _barrier is None:
        raise RuntimeError("test barrier is not configured")
    _barrier.wait(timeout=5)
    return value


def fail() -> None:
    raise RuntimeError("boom")


def model_spec(**declarations: object) -> ModelSpec:
    return ModelSpec.model_validate(
        {"schema_version": 2, "sources": {}, **declarations}
    )


def test_empty_preprocessing_returns_no_supplied_values(prefect_server) -> None:
    assert preprocess({"source": object()}, model_spec()) == {}


def test_preprocessing_returns_only_declared_results(prefect_server) -> None:
    spec = model_spec(
        preprocessing={
            "squared": {
                "call": "builtins.pow",
                "kwargs": {"base": Ref("value"), "exp": 2},
            },
            "ratio": {"call": Ref("squared.as_integer_ratio")},
        }
    )

    result = preprocess({"value": 3, "unused": object()}, spec)

    assert result == {"squared": 9, "ratio": (9, 1)}
    with get_client(sync_client=True) as client:
        runs = client.read_task_runs(
            task_run_filter=TaskRunFilter(
                name=TaskRunFilterName(
                    any_=["preprocess-squared", "preprocess-ratio"]
                )
            )
        )
    assert {run.name for run in runs} == {
        "preprocess-squared",
        "preprocess-ratio",
    }


def test_preprocessing_validates_all_inputs_before_submission(monkeypatch) -> None:
    spec = model_spec(
        preprocessing={
            "first": {"call": "builtins.dict"},
            "second": {"call": Ref("missing")},
        }
    )

    def unexpected_configuration(**kwargs: object) -> None:
        pytest.fail(f"task configured before validation: {kwargs}")

    monkeypatch.setattr(invoke_call, "with_options", unexpected_configuration)

    with pytest.raises(ValueError, match="missing"):
        preprocess.fn({}, spec)


def test_preprocessing_rejects_forward_references_before_submission(
    monkeypatch,
) -> None:
    spec = model_spec(
        preprocessing={
            "first": {"call": Ref("second")},
            "second": {"call": "builtins.dict"},
        }
    )

    def unexpected_configuration(**kwargs: object) -> None:
        pytest.fail(f"task configured before validation: {kwargs}")

    monkeypatch.setattr(invoke_call, "with_options", unexpected_configuration)

    with pytest.raises(ValueError, match="Forward.*second"):
        preprocess.fn({}, spec)


def test_source_validation_runs_before_submission(monkeypatch) -> None:
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {"optical": {"variables": ["red"]}},
            "preprocessing": {
                "result": {
                    "call": "builtins.dict",
                    "kwargs": {"image": Ref("optical")},
                }
            },
        }
    )

    def unexpected_configuration(**kwargs: object) -> None:
        pytest.fail(f"task configured before source validation: {kwargs}")

    monkeypatch.setattr(invoke_call, "with_options", unexpected_configuration)

    with pytest.raises(ValueError, match="red"):
        preprocess.fn({"optical": xr.Dataset()}, spec)


@pytest.mark.parametrize(
    ("selector", "names"),
    [
        ({"variables": ["nir", "red"]}, ["nir", "red"]),
        ({"channels": 2}, ["red", "nir"]),
    ],
)
def test_preprocessing_selects_only_consumed_sources(
    raw, prefect_server, selector: dict[str, object], names: list[str]
) -> None:
    source = raw["optical"]
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {
                "optical": selector,
                "unused": {"variables": ["missing"]},
            },
            "preprocessing": {
                "result": {
                    "call": "builtins.dict",
                    "kwargs": {"image": Ref("optical")},
                }
            },
        }
    )
    computed: list[object] = []

    with Callback(pretask=lambda *args: computed.append(args)):
        result = preprocess({"optical": source}, spec)

    selected = result["result"]["image"]
    assert computed == []
    assert list(selected.data_vars) == names
    assert selected.red.data is source.red.data
    assert list(source.data_vars) == ["red", "nir", "unused"]


def test_preprocessing_rebinds_from_the_supplied_value(prefect_server) -> None:
    spec = model_spec(
        preprocessing={
            "value": {
                "call": "builtins.pow",
                "kwargs": {"base": Ref("value"), "exp": 2},
            }
        }
    )

    assert preprocess({"value": 3}, spec) == {"value": 9}


def test_preprocessing_preserves_none_results(prefect_server) -> None:
    items: dict[str, int] = {}
    spec = model_spec(
        preprocessing={
            "updated": {
                "call": Ref("items.update"),
                "kwargs": {"answer": 42},
            }
        }
    )

    assert preprocess({"items": items}, spec) == {"updated": None}
    assert items == {"answer": 42}


def test_inactive_inference_import_is_never_loaded(prefect_server) -> None:
    spec = model_spec(
        preprocessing={"result": {"call": "builtins.dict"}},
        inference={"image": {"call": "missing_package.to_tensor"}},
    )

    assert preprocess({}, spec) == {"result": {}}


def test_independent_calls_can_overlap(prefect_server) -> None:
    global _barrier
    _barrier = Barrier(2)
    spec = model_spec(
        preprocessing={
            "first": {"call": f"{__name__}.meet", "kwargs": {"value": 1}},
            "second": {"call": f"{__name__}.meet", "kwargs": {"value": 2}},
        }
    )
    try:
        assert preprocess({}, spec) == {"first": 1, "second": 2}
    finally:
        _barrier = None


def test_failed_dependency_keeps_declaration_names(prefect_server) -> None:
    spec = model_spec(
        preprocessing={
            "failed_call": {"call": f"{__name__}.fail"},
            "dependent_call": {
                "call": "builtins.dict",
                "kwargs": {"value": Ref("failed_call")},
            },
            "independent_call": {
                "call": "builtins.dict",
                "kwargs": {"value": 1},
            },
        }
    )

    with pytest.raises(RuntimeError, match="boom"):
        preprocess({}, spec)

    with get_client(sync_client=True) as client:
        runs = client.read_task_runs()

    states = {
        declaration: next(
            run.state_name
            for run in runs
            if run.name.startswith(f"preprocess-{declaration}")
        )
        for declaration in ("failed_call", "dependent_call", "independent_call")
    }
    assert states == {
        "failed_call": "Failed",
        "dependent_call": "NotReady",
        "independent_call": "Completed",
    }
