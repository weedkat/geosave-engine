from __future__ import annotations

import math

from pydantic import ValidationError
import pytest

from geosave_engine.model.spec import CallSpec, Ref


def mutate(*, items: list[int]) -> list[int]:
    items.append(2)
    return items


def test_call_invokes_imports_bound_methods_and_nested_references() -> None:
    imported = CallSpec.model_validate(
        {
            "call": "builtins.dict",
            "kwargs": {
                "nested": [Ref("source")],
                "literal": {"ref": "source"},
                "text": "!ref source",
            },
        }
    )
    source = object()

    assert imported.invoke({"source": source}) == {
        "nested": [source],
        "literal": {"ref": "source"},
        "text": "!ref source",
    }
    assert CallSpec(call=Ref("number.as_integer_ratio")).invoke(
        {"number": 1.5}
    ) == (3, 2)


def test_call_selects_only_referenced_roots() -> None:
    spec = CallSpec(
        call=Ref("function"),
        kwargs={"left": Ref("wait_for"), "right": Ref("return_state")},
    )

    selected = spec.select_inputs(
        {
            "function": lambda **kwargs: kwargs,
            "wait_for": 1,
            "return_state": 2,
            "unused": object(),
        }
    )

    assert set(selected) == {"function", "wait_for", "return_state"}


def test_call_rejects_missing_inputs_before_invocation() -> None:
    spec = CallSpec(call="builtins.dict", kwargs={"value": Ref("missing")})

    with pytest.raises(ValueError, match="missing"):
        spec.select_inputs({"unused": object()})


def test_call_rebuilds_literal_containers() -> None:
    spec = CallSpec(call=Ref("mutate"), kwargs={"items": [1]})

    first = spec.invoke({"mutate": mutate})
    second = spec.invoke({"mutate": mutate})

    assert first == second == [1, 2]
    assert first is not second


def test_call_binds_none_as_the_complete_result() -> None:
    items: dict[str, int] = {}
    spec = CallSpec(call=Ref("items.update"), kwargs={"answer": 42})

    assert spec.invoke({"items": items}) is None
    assert items == {"answer": 42}


@pytest.mark.parametrize("path", ["", "a..b", ".a", "a.", "a[0]", "a()"])
def test_reference_rejects_non_attribute_paths(path: str) -> None:
    with pytest.raises(ValueError):
        Ref(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": float("nan")},
        {"value": math.inf},
        {1: "value"},
        {"value": object()},
    ],
)
def test_call_rejects_non_yaml_keyword_values(kwargs: object) -> None:
    with pytest.raises((TypeError, ValueError, ValidationError)):
        CallSpec.model_validate({"call": "builtins.dict", "kwargs": kwargs})


def test_call_rejects_cyclic_keyword_values() -> None:
    kwargs: dict[str, object] = {}
    kwargs["self"] = kwargs

    with pytest.raises((ValueError, ValidationError), match="Cyclic|recursion"):
        CallSpec.model_validate({"call": "builtins.dict", "kwargs": kwargs})


@pytest.mark.parametrize("call", ["missing_package.function", "math.pi"])
def test_call_defers_invalid_active_imports_until_invocation(call: str) -> None:
    spec = CallSpec(call=call)

    with pytest.raises((ImportError, TypeError)):
        spec.invoke({})


def test_call_validates_signature_before_calling() -> None:
    called: list[object] = []

    def function(*, expected: object) -> None:
        called.append(expected)

    spec = CallSpec(call=Ref("function"), kwargs={"wrong": 1})

    with pytest.raises(TypeError):
        spec.invoke({"function": function})

    assert called == []


def test_call_rejects_missing_and_noncallable_attributes() -> None:
    class Source:
        value = 1

    with pytest.raises(TypeError, match="callable"):
        CallSpec(call=Ref("source.value")).invoke({"source": Source()})
    with pytest.raises(AttributeError):
        CallSpec(call=Ref("source.missing")).invoke({"source": Source()})


def test_call_error_note_identifies_the_declared_target() -> None:
    def fail() -> None:
        raise RuntimeError("failed")

    spec = CallSpec(call=Ref("fail"))

    with pytest.raises(RuntimeError) as caught:
        spec.invoke({"fail": fail})

    assert caught.value.__notes__ == ["While invoking Ref('fail')"]
