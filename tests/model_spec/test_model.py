from __future__ import annotations

import inspect

import pytest
import yaml

from geosave_engine.model_spec import CallSpec, ModelSpec, Ref


@pytest.mark.parametrize(
    "field",
    ["tiling", "model_inputs", "aggregation", "postprocessing", "exports"],
)
def test_prediction_fields_are_rejected(field):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ModelSpec.model_validate(
            {"schema_version": 2, "rasters": {}, field: {}}
        )


def test_resolve_path_is_static_and_uses_model_filename(tmp_path):
    descriptor = inspect.getattr_static(ModelSpec, "resolve_path")

    assert isinstance(descriptor, staticmethod)
    assert ModelSpec.filename == "model_spec.yaml"
    assert ModelSpec.resolve_path(tmp_path) == tmp_path / ModelSpec.filename


def test_python_and_yaml_round_trip_preserves_references_and_literals(tmp_path):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"channels": 1}},
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
        }
    )
    assert ModelSpec.model_validate(spec.model_dump()) == spec
    (tmp_path / "weights.bin").write_bytes(b"weights")

    path = spec.save(tmp_path)

    assert "!ref" in path.read_text()
    assert ModelSpec.load(path) == spec
    assert ModelSpec.load(tmp_path) == spec
    assert (tmp_path / "weights.bin").read_bytes() == b"weights"


def test_parsing_never_imports_or_executes_calls(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 2
rasters: {source: {channels: 1}}
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
        "schema_version: 2\nrasters: {}\nrasters: {}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, call: builtins.list}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: !ref [model]}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: !unknown model}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {!ref key: 1}}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {1: value}}",
        "schema_version: 2\nrasters: {1: {}}",
        "schema_version: 2\nrasters: {}\npreprocessing: {1: {call: builtins.dict}}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: &cycle {self: *cycle}}",
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
        ModelSpec.model_validate({"schema_version": 2, "rasters": {}, **change})


def test_schema_version_is_explicit():
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"rasters": {}})


def test_old_sources_field_is_rejected():
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"schema_version": 2, "sources": {}})


def test_yaml_reference_support_does_not_modify_safe_loader():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("value: !ref input")


def test_save_revalidates_mutated_declarations(tmp_path):
    spec = ModelSpec(schema_version=2, rasters={})
    spec.preprocessing.root["bad.name"] = CallSpec(call="builtins.dict")

    with pytest.raises(ValueError):
        spec.save(tmp_path)

    assert not (tmp_path / "model_spec.yaml").exists()
