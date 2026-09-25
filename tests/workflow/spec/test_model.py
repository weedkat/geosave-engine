from __future__ import annotations

import pytest
import yaml

from geosave_engine.geodata import attrs
from geosave_engine.workflow.spec import ModelSpec, OperationSpec, Ref


@pytest.mark.parametrize("color_map", [None, {0: "#000000"}])
def test_output_legends_round_trip_python_and_yaml(tmp_path, color_map):
    from geosave_engine.workflow.spec import OutputSpec

    legend = attrs.Legend(class_map={0: "background"}, color_map=color_map)
    spec = ModelSpec(
        schema_version=2,
        sources={},
        outputs={"prediction": OutputSpec(legend=legend)},
    )
    restored = ModelSpec.model_validate(spec.model_dump())
    assert restored.outputs["prediction"].legend == legend
    assert ModelSpec.load(spec.save(tmp_path)).outputs["prediction"].legend == legend


def test_output_legend_loads_native_integer_class_and_color_keys(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("""schema_version: 2
sources: {}
outputs:
  prediction:
    legend:
      class_map: {0: background}
      color_map: {0: '#000000'}
""")
    assert ModelSpec.load(path).outputs["prediction"].legend == attrs.Legend(
        class_map={0: "background"}, color_map={0: "#000000"}
    )


def test_python_and_yaml_round_trip_preserves_references_and_literals(tmp_path):
    spec = ModelSpec(
        schema_version=2,
        sources={},
        preprocessing={
            "selected": OperationSpec(
                call=Ref("optical.__getitem__"),
                kwargs={"key": ["red", "nir"]},
            ),
            "result": OperationSpec(
                call="missing_package.for_this_stage",
                kwargs={
                    "items": [Ref("selected"), {"ref": "literal"}],
                    "label": "!ref selected",
                },
            ),
        },
        inference={"logits": {"call": Ref("model"), "kwargs": {"image": Ref("image")}}},
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
    path.write_text("""schema_version: 2
sources: {}
preprocessing:
  result:
    call: deliberately_uninstalled.module.function
    kwargs: {input: !ref source}
""")
    assert ModelSpec.load(path).preprocessing["result"].kwargs["input"] == Ref("source")


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
        "schema_version: 2\nsources: {}\noutputs: {1: {}}",
        "schema_version: 2\nsources: {}\noutputs:\n  prediction:\n    legend:\n      color_map: {0: '#000000', 0: '#ffffff'}",
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
        ModelSpec(sources={})


def test_yaml_reference_support_does_not_modify_safe_loader():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("value: !ref input")


def test_save_revalidates_mutated_declarations(tmp_path):
    spec = ModelSpec(schema_version=2, sources={})
    spec.preprocessing["bad.name"] = OperationSpec(call="builtins.dict")
    with pytest.raises(ValueError):
        spec.save(tmp_path)
    assert not (tmp_path / "model_spec.yaml").exists()
