from __future__ import annotations

import pytest
from pydantic import ValidationError
import yaml

from geosave_engine.workflow.specs import CallSpec, ModelSpec, Ref, StageSpec


def test_inference_reuses_explicit_call_declarations(tmp_path):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {},
            "preprocessing": {},
            "inference": {
                "image": {
                    "call": Ref("normalized.gs.to_tensor"),
                    "kwargs": {"dtype": "float32"},
                }
            },
            "postprocessing": {},
        }
    )

    path = spec.save(tmp_path)

    assert isinstance(spec.preprocessing, StageSpec)
    assert isinstance(spec.inference, StageSpec)
    assert ModelSpec.load(path) == spec


def test_postprocessing_rejects_undeclared_semantics():
    with pytest.raises(ValidationError, match="method"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "sources": {},
                "postprocessing": {"method": "segmentation"},
            }
        )


def test_loading_never_imports_inference_calls(tmp_path):
    path = tmp_path / "model_spec.yaml"
    path.write_text(
        "schema_version: 2\nsources: {}\ninference:\n"
        "  image:\n    call: missing_package.to_tensor\n"
        "postprocessing: {}\n"
    )

    assert ModelSpec.load(path).inference["image"].call == ("missing_package.to_tensor")


def test_python_and_yaml_round_trip_preserves_references_and_literals(tmp_path):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {},
            "preprocessing": {
                "selected": CallSpec(
                    call=Ref("optical.__getitem__"),
                    kwargs={"key": ["red", "nir"]},
                ),
                "result": CallSpec(
                    call="missing_package.for_this_stage",
                    kwargs={
                        "items": [Ref("selected"), {"ref": "literal"}],
                        "label": "!ref selected",
                    },
                ),
            },
            "inference": {
                "logits": {
                    "call": Ref("model"),
                    "kwargs": {"image": Ref("image")},
                },
            },
        }
    )
    assert ModelSpec.model_validate(spec.model_dump()) == spec
    (tmp_path / "weights.bin").write_bytes(b"weights")

    path = spec.save(tmp_path)

    assert "!ref" in path.read_text()
    assert ModelSpec.load(path) == spec
    assert ModelSpec.load(tmp_path) == spec
    assert (tmp_path / "weights.bin").read_bytes() == b"weights"
    assert isinstance(ModelSpec.load(path).inference["logits"].call, Ref)


def test_parsing_never_imports_or_executes_calls(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 2
sources: {}
preprocessing:
  result:
    call: deliberately_uninstalled.module.function
    kwargs: {input: !ref source}
"""
    )

    result = ModelSpec.load(path)

    assert result.preprocessing["result"].kwargs["input"] == Ref("source")


@pytest.mark.parametrize(
    "path", ["", "a..b", ".a", "a.", "a[0]", "a()", "two names", "3a"]
)
def test_invalid_reference_paths_fail(path):
    with pytest.raises(ValueError):
        Ref(path)


@pytest.mark.parametrize(
    "body",
    [
        "schema_version: 2\nsources: {}\nsources: {}",
        "schema_version: 2\nsources: {}\npreprocessing:\n  x: {call: builtins.dict, call: builtins.list}",
        "schema_version: 2\nsources: {}\npreprocessing:\n  x: {call: !ref [model]}",
        "schema_version: 2\nsources: {}\npreprocessing:\n  x: {call: !unknown model}",
        "schema_version: 2\nsources: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {!ref key: 1}}",
        "schema_version: 2\nsources: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {1: value}}",
        "schema_version: 2\nsources: {1: {}}",
        "schema_version: 2\nsources: {}\npreprocessing: {1: {call: builtins.dict}}",
        "schema_version: 2\nsources: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: &cycle {self: *cycle}}",
    ],
)
def test_invalid_yaml_is_rejected(tmp_path, body):
    path = tmp_path / "model.yaml"
    path.write_text(body)

    with pytest.raises((ValueError, yaml.YAMLError)):
        ModelSpec.load(path)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 1},
        {"schema_version": 3},
        {"extra": {}},
        {"preprocessing": {"bad.name": {"call": "builtins.dict"}}},
        {"preprocessing": {"value": {"call": "builtins.dict", "references": {}}}},
        {
            "preprocessing": {
                "value": {"call": "builtins.dict", "kwargs": {"x": float("nan")}}
            }
        },
        {
            "preprocessing": {
                "value": {"call": "builtins.dict", "kwargs": {"x": object()}}
            }
        },
    ],
)
def test_invalid_python_declarations_fail(change):
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"schema_version": 2, "sources": {}, **change})


def test_schema_version_is_explicit():
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"sources": {}})


def test_yaml_reference_support_does_not_modify_safe_loader():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("value: !ref input")


def test_save_revalidates_mutated_declarations(tmp_path):
    spec = ModelSpec(schema_version=2, sources={})
    spec.preprocessing.root["bad.name"] = CallSpec(call="builtins.dict")

    with pytest.raises(ValueError):
        spec.save(tmp_path)

    assert not (tmp_path / "model_spec.yaml").exists()
