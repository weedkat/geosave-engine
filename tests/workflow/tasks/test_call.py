from prefect import Task
from prefect.cache_policies import NO_CACHE

from geosave_engine.workflow.specs import CallSpec, Ref
from geosave_engine.workflow.tasks import invoke_call


def test_invoke_call_is_one_nonpersisted_uncached_task() -> None:
    assert isinstance(invoke_call, Task)
    assert invoke_call.name == "model-call"
    assert invoke_call.cache_policy is NO_CACHE
    assert invoke_call.persist_result is False


def test_invoke_call_delegates_to_the_spec() -> None:
    spec = CallSpec(
        call="builtins.pow",
        kwargs={"base": Ref("value"), "exp": 2},
    )

    assert invoke_call.fn(spec, {"value": 3}) == 9


def test_prefect_reserved_names_are_positional_runtime_inputs() -> None:
    spec = CallSpec(
        call="builtins.dict",
        kwargs={
            "wait_for": Ref("wait_for"),
            "return_state": Ref("return_state"),
        },
    )

    assert invoke_call.fn(spec, {"wait_for": 1, "return_state": 2}) == {
        "wait_for": 1,
        "return_state": 2,
    }
