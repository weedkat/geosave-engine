from types import ModuleType
import sys

from dask.callbacks import Callback
import pytest
import xarray as xr

from geosave_engine.workflow.specs import ModelSpec, Ref
from geosave_engine.workflow.tasks.preprocess import Preprocessor, preprocess


def model_spec(**declarations):
    return ModelSpec(schema_version=2, sources={}, **declarations)


def test_preprocess_keeps_native_lazy_values(raw):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {},
            "preprocessing": {
                "selected": {
                    "call": Ref("optical.__getitem__"),
                    "kwargs": {"key": ["nir", "red"]},
                }
            },
            "inference": {"image": {"call": "missing_package.to_tensor"}},
            "postprocessing": {},
        }
    )

    result = preprocess.fn(raw, spec)

    assert list(result["selected"].data_vars) == ["nir", "red"]
    assert result["selected"].nir.chunks is not None


def test_preprocess_validates_all_inputs_before_the_first_call():
    called = []
    spec = model_spec(
        preprocessing={
            "first": {"call": Ref("record"), "kwargs": {}},
            "second": {"call": Ref("missing"), "kwargs": {}},
        }
    )

    with pytest.raises(ValueError, match="missing"):
        preprocess.fn({"record": lambda: called.append(True)}, spec)

    assert called == []


def test_inactive_inference_import_is_never_loaded():
    spec = model_spec(
        preprocessing={"result": {"call": "builtins.dict"}},
        inference={"image": {"call": "missing_package.to_tensor"}},
    )

    assert Preprocessor(spec).run({})["result"] == {}


def test_nested_references_and_literal_strings_are_unambiguous():
    opaque = {"value": Ref("not_configuration")}
    processor = Preprocessor(
        model_spec(
            preprocessing={
                "result": {
                    "call": "builtins.dict",
                    "kwargs": {
                        "nested": [Ref("opaque")],
                        "literal": {"ref": "opaque"},
                        "text": "!ref opaque",
                    },
                }
            }
        )
    )

    result = processor.run({"opaque": opaque})["result"]

    assert result["nested"][0] is opaque
    assert result["literal"] == {"ref": "opaque"}
    assert result["text"] == "!ref opaque"


def test_rebinding_uses_the_previous_value_and_preserves_aliases():
    source = xr.Dataset({"red": ("x", [1, 2]), "nir": ("x", [3, 4])})
    processor = Preprocessor(
        model_spec(
            preprocessing={
                "optical": {
                    "call": Ref("optical.__getitem__"),
                    "kwargs": {"key": ["red"]},
                }
            }
        )
    )
    supplied = {"optical": source, "alias": source}

    result = processor.run(supplied)

    assert list(result["optical"].data_vars) == ["red"]
    assert result["alias"] is source
    assert supplied["optical"] is source
    assert list(source.data_vars) == ["red", "nir"]


def test_mutating_call_binds_none_and_keeps_explicit_state():
    items = {}
    processor = Preprocessor(
        model_spec(
            preprocessing={
                "updated": {"call": Ref("items.update"), "kwargs": {"answer": 42}}
            }
        )
    )

    result = processor.run({"items": items})

    assert result["updated"] is None
    assert result["items"] is items
    assert items == {"answer": 42}


def test_repeated_runs_use_fresh_literal_containers_and_bindings():
    def mutate(values):
        values.append(2)
        return values

    processor = Preprocessor(
        model_spec(
            preprocessing={"result": {"call": Ref("mutate"), "kwargs": {"values": [1]}}}
        )
    )

    first = processor.run({"mutate": mutate})
    second = processor.run({"mutate": mutate})

    assert first["result"] == second["result"] == [1, 2]
    assert first is not second
    assert first["result"] is not second["result"]


def test_preprocessor_captures_a_validated_configuration_copy():
    spec = model_spec(
        preprocessing={"result": {"call": "builtins.dict", "kwargs": {"value": 1}}}
    )
    processor = Preprocessor(spec)
    spec.preprocessing["result"].kwargs["value"] = 99

    assert processor.run({})["result"] == {"value": 1}


def test_source_validation_runs_before_any_call():
    called = []
    spec = ModelSpec(
        schema_version=2,
        sources={"optical": {"variables": ["red"]}},
        preprocessing={
            "result": {"call": Ref("mark"), "kwargs": {"data": Ref("optical")}}
        },
    )

    with pytest.raises(ValueError, match="red"):
        Preprocessor(spec).run(
            {"mark": lambda data: called.append(True), "optical": xr.Dataset()}
        )

    assert called == []


@pytest.mark.parametrize(
    ("selector", "names"),
    [
        ({"variables": ["nir", "red"]}, ["nir", "red"]),
        ({"channels": 2}, ["red", "nir"]),
    ],
)
def test_preprocessor_selects_only_consumed_sources(raw, selector, names):
    source = raw["optical"]
    spec = ModelSpec(
        schema_version=2,
        sources={"optical": selector, "unused": {"variables": ["missing"]}},
        preprocessing={
            "result": {"call": "builtins.dict", "kwargs": {"image": Ref("optical")}}
        },
    )
    computed = []

    with Callback(pretask=lambda *args: computed.append(True)):
        result = Preprocessor(spec).run({"optical": source})

    selected = result["result"]["image"]
    assert computed == []
    assert selected is result["optical"]
    assert list(selected.data_vars) == names
    assert selected.red.data is source.red.data
    assert list(source.data_vars) == ["red", "nir", "unused"]


def test_empty_preprocessing_is_a_fresh_noop():
    source = object()
    supplied = {"source": source}

    result = Preprocessor(model_spec()).run(supplied)

    assert result == supplied
    assert result is not supplied
    assert result["source"] is source


@pytest.mark.parametrize("call", ["uninstalled_active.function", "math.pi"])
def test_invalid_active_import_fails_when_preprocessor_is_built(call):
    with pytest.raises((ImportError, TypeError, ValueError)):
        Preprocessor(model_spec(preprocessing={"result": {"call": call}}))


def test_imported_signature_error_fails_before_execution():
    with pytest.raises(TypeError, match="required argument"):
        Preprocessor(
            model_spec(
                preprocessing={"result": {"call": "math.sqrt", "kwargs": {"wrong": 1}}}
            )
        )


def test_imported_callable_containers_are_not_traversed(monkeypatch):
    class Scale(dict):
        def __call__(self, *, value):
            return self["factor"] * value

    module = ModuleType("example_operations")
    module.scale = Scale(factor=3)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    processor = Preprocessor(
        model_spec(
            preprocessing={
                "result": {
                    "call": "example_operations.scale",
                    "kwargs": {"value": 2},
                }
            }
        )
    )

    assert processor.run({})["result"] == 6


def test_bound_signature_error_does_not_call_target():
    called = []

    def function(*, expected):
        called.append(expected)

    processor = Preprocessor(
        model_spec(
            preprocessing={"result": {"call": Ref("function"), "kwargs": {"wrong": 1}}}
        )
    )

    with pytest.raises(TypeError):
        processor.run({"function": function})

    assert called == []


def test_bound_noncallable_and_missing_attribute_fail_clearly():
    processor = Preprocessor(
        model_spec(preprocessing={"result": {"call": Ref("source.value")}})
    )

    class Source:
        value = 1

    with pytest.raises(TypeError, match="callable"):
        processor.run({"source": Source()})
    with pytest.raises(AttributeError):
        processor.run({"source": object()})


def test_preprocess_task_does_not_cache_or_persist_native_results():
    assert preprocess.cache_policy is None
    assert preprocess.persist_result is False
