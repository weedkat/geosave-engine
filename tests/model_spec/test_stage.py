from __future__ import annotations

from collections.abc import Mapping

import pytest

from geosave_engine.model_spec import Ref, StageSpec


def stage(**calls: object) -> StageSpec:
    return StageSpec.model_validate(calls)


def test_stage_is_an_ordered_mapping_with_plain_dump_shape() -> None:
    spec = stage(
        first={"call": "builtins.dict"},
        second={"call": "builtins.list"},
    )

    assert isinstance(spec, Mapping)
    assert list(spec) == ["first", "second"]
    assert spec["first"].call == "builtins.dict"
    assert spec.model_dump() == {
        "first": {"call": "builtins.dict", "kwargs": {}},
        "second": {"call": "builtins.list", "kwargs": {}},
    }


def test_stage_validates_external_inputs_and_order() -> None:
    spec = stage(
        selected={"call": Ref("source.select")},
        result={"call": "builtins.dict", "kwargs": {"value": Ref("selected")}},
    )

    assert spec.external_inputs == ("source",)
    assert spec.validate_inputs({"source", "unused"}) == frozenset({"source"})


def test_stage_rejects_missing_and_forward_references() -> None:
    missing = stage(result={"call": Ref("source.method")})
    forward = stage(
        first={"call": Ref("second")},
        second={"call": "builtins.dict"},
    )

    with pytest.raises(ValueError, match="source"):
        missing.validate_inputs(set())
    with pytest.raises(ValueError, match="Forward.*second"):
        forward.validate_inputs(set())

    assert forward.validate_inputs({"second"}) == frozenset({"second"})


def test_stage_allows_explicit_rebinding() -> None:
    spec = stage(
        value={
            "call": Ref("scale"),
            "kwargs": {"value": Ref("value"), "factor": 3},
        }
    )

    assert spec.external_inputs == ("scale", "value")
    assert spec.validate_inputs({"scale", "value"}) == frozenset(
        {"scale", "value"}
    )


def test_empty_stage_has_no_inputs() -> None:
    spec = StageSpec.model_validate({})

    assert spec.external_inputs == ()
    assert spec.validate_inputs(set()) == frozenset()
