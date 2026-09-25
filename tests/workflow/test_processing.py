from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import dask.array as da
import numpy as np
from dask.callbacks import Callback
import pytest
import xarray as xr

from geosave_engine.geodata.transform.tiling import Tiles
from geosave_engine.workflow.processing import Processor
from geosave_engine.workflow.spec import ModelSpec, Ref


def spec(**stages):
    return ModelSpec(schema_version=2, sources={}, **stages)


def test_preprocessing_loads_alone_and_preserves_native_lazy_values(tmp_path):
    source = xr.Dataset(
        {
            "red": (("y", "x"), da.ones((4, 4), chunks=(2, 2))),
            "nir": (("y", "x"), da.full((4, 4), 2.0, chunks=(2, 2))),
            "extra": (("y", "x"), da.zeros((4, 4), chunks=(2, 2))),
        }
    )
    configuration = ModelSpec(
        schema_version=2,
        sources={
            "optical": {
                "type": "raster",
                "variables": ["red", "nir"],
                "dims": ["y", "x"],
            },
            "unused": {"variables": ["absent"]},
        },
        preprocessing={
            "selected": {
                "call": Ref("optical.__getitem__"),
                "kwargs": {"key": ["nir", "red"]},
            },
            "reflectance": {"call": Ref("selected.gs.to_nan")},
        },
        inference={"logits": {"call": "uninstalled_inference.predict"}},
        postprocessing={"result": {"call": "uninstalled_postprocessing.process"}},
    )
    processor = Processor.load(stage="preprocessing", path=configuration.save(tmp_path))
    supplied = {"optical": source}
    computed = []
    with Callback(posttask=lambda *args: computed.append(True)):
        result = processor(supplied)
    assert computed == []
    assert list(result["optical"].data_vars) == ["red", "nir"]
    assert result["optical"].red.data is source.red.data
    assert list(result["reflectance"].data_vars) == ["nir", "red"]
    assert isinstance(result["reflectance"].nir.data, da.Array)
    assert list(source.data_vars) == ["red", "nir", "extra"]
    assert set(supplied) == {"optical"}


def test_postprocessing_needs_only_its_own_supplied_values(tmp_path):
    configuration = spec(
        preprocessing={"prepared": {"call": "uninstalled_preprocessing.prepare"}},
        postprocessing={
            "result": {"call": Ref("finish"), "kwargs": {"value": Ref("logits")}}
        },
    )
    finish = Processor.load(stage="postprocessing", path=configuration.save(tmp_path))
    result = finish({"finish": lambda value: value + 2, "logits": 3})
    assert result["result"] == 5


def test_resolved_objects_are_passed_unchanged_and_not_interpreted_again():
    opaque = {"value": Ref("not_configuration")}
    prepare = Processor(
        stage="preprocessing",
        spec=spec(
            preprocessing={
                "result": {
                    "call": "builtins.dict",
                    "kwargs": {"nested": [Ref("opaque")], "literal": {"ref": "opaque"}},
                },
            }
        ),
    )
    result = prepare({"opaque": opaque})["result"]
    assert result["nested"][0] is opaque
    assert result["literal"] == {"ref": "opaque"}


def test_rebinding_resolves_the_previous_value_and_preserves_aliases():
    source = xr.Dataset({"red": ("x", [1, 2]), "nir": ("x", [3, 4])})
    prepare = Processor(
        stage="preprocessing",
        spec=spec(
            preprocessing={
                "optical": {
                    "call": Ref("optical.__getitem__"),
                    "kwargs": {"key": ["red"]},
                },
            }
        ),
    )
    values = {"optical": source, "alias": source}
    result = prepare(values)
    assert list(result["optical"].data_vars) == ["red"]
    assert result["alias"] is source
    assert values["optical"] is source
    assert list(source.data_vars) == ["red", "nir"]


def test_mutating_calls_bind_actual_none_and_keep_explicit_state():
    items = {}
    prepare = Processor(
        stage="preprocessing",
        spec=spec(
            preprocessing={
                "updated": {"call": Ref("items.update"), "kwargs": {"answer": 42}},
            }
        ),
    )
    result = prepare({"items": items})
    assert result["updated"] is None
    assert result["items"] is items
    assert items == {"answer": 42}


def test_each_call_has_fresh_literal_containers_and_name_bindings():
    def mutate(values):
        values.append(2)
        return values

    prepare = Processor(
        stage="preprocessing",
        spec=spec(
            preprocessing={
                "result": {"call": Ref("mutate"), "kwargs": {"values": [1]}},
            }
        ),
    )
    first = prepare({"mutate": mutate})
    second = prepare({"mutate": mutate})
    assert first["result"] == second["result"] == [1, 2]
    assert first is not second and first["result"] is not second["result"]


def test_processor_captures_configuration_without_sharing_nested_dicts():
    configuration = spec(
        preprocessing={"result": {"call": "builtins.dict", "kwargs": {"value": 1}}}
    )
    prepare = Processor(stage="preprocessing", spec=configuration)
    configuration.preprocessing["result"].kwargs["value"] = 99
    assert prepare({})["result"] == {"value": 1}


def test_missing_later_reference_fails_before_any_call():
    called = []
    prepare = Processor(
        stage="preprocessing",
        spec=spec(
            preprocessing={
                "first": {"call": Ref("mark")},
                "last": {"call": "builtins.dict", "kwargs": {"value": Ref("missing")}},
            }
        ),
    )
    with pytest.raises(ValueError, match="missing"):
        prepare({"mark": lambda: called.append(True)})
    assert called == []


def test_unbound_self_reference_fails():
    prepare = Processor(
        stage="preprocessing",
        spec=spec(preprocessing={"value": {"call": Ref("value")}}),
    )
    with pytest.raises(ValueError, match="value"):
        prepare({})


def test_source_validation_runs_before_work():
    called = []
    configuration = ModelSpec(
        schema_version=2,
        sources={"optical": {"variables": ["red"]}},
        preprocessing={
            "result": {"call": Ref("mark"), "kwargs": {"data": Ref("optical")}}
        },
    )
    with pytest.raises(ValueError, match="red"):
        Processor(stage="preprocessing", spec=configuration)(
            {"mark": lambda data: called.append(True), "optical": xr.Dataset()}
        )
    assert called == []


@pytest.mark.parametrize(
    "selector,names",
    [
        ({"variables": ["nir", "red"]}, ["nir", "red"]),
        ({"channels": 2}, ["red", "nir"]),
    ],
)
def test_processor_binds_selected_sources_before_operations(raw, selector, names):
    source = raw["optical"]
    configuration = ModelSpec(
        schema_version=2,
        sources={"optical": selector, "unused": {"variables": ["missing"]}},
        preprocessing={
            "result": {"call": "builtins.dict", "kwargs": {"image": Ref("optical")}}
        },
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = Processor(configuration, stage="preprocessing")({"optical": source})
    selected = result["result"]["image"]
    assert tasks == []
    assert selected is result["optical"]
    assert list(selected.data_vars) == names
    assert selected.red.data is source.red.data
    assert list(source.data_vars) == ["red", "nir", "unused"]


def test_empty_stage_is_an_independent_noop():
    source = object()
    original = {"source": source}
    result = Processor(stage="preprocessing", spec=spec())(original)
    assert result == original and result is not original
    assert result["source"] is source


@pytest.mark.parametrize("call", ["uninstalled_active.function", "math.pi"])
def test_invalid_active_imports_fail_when_processor_is_built(call):
    with pytest.raises((ImportError, TypeError, ValueError)):
        Processor(
            stage="preprocessing", spec=spec(preprocessing={"result": {"call": call}})
        )


def test_imported_signature_errors_fail_before_execution():
    with pytest.raises(TypeError, match="required argument"):
        Processor(
            stage="preprocessing",
            spec=spec(
                preprocessing={"result": {"call": "math.sqrt", "kwargs": {"wrong": 1}}}
            ),
        )


def test_imported_callable_containers_are_not_traversed(monkeypatch):
    class Scale(dict):
        def __call__(self, *, value):
            return self["factor"] * value

    module = ModuleType("example_operations")
    module.scale = Scale(factor=3)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    prepare = Processor(
        spec(
            preprocessing={
                "result": {"call": "example_operations.scale", "kwargs": {"value": 2}}
            }
        ),
        stage="preprocessing",
    )
    assert prepare({})["result"] == 6


def test_bound_signature_errors_do_not_call_target():
    called = []

    def function(*, expected):
        called.append(expected)

    prepare = Processor(
        stage="preprocessing",
        spec=spec(
            preprocessing={"result": {"call": Ref("function"), "kwargs": {"wrong": 1}}}
        ),
    )
    with pytest.raises(TypeError):
        prepare({"function": function})
    assert called == []


def test_bound_noncallable_and_missing_attribute_fail_clearly():
    prepare = Processor(
        stage="preprocessing",
        spec=spec(preprocessing={"result": {"call": Ref("source.value")}}),
    )

    class Source:
        value = 1

    with pytest.raises(TypeError, match="callable"):
        prepare({"source": Source()})
    with pytest.raises(AttributeError):
        prepare({"source": object()})


def test_core_imports_do_not_import_flows_prefect_or_lightning():
    script = """
import sys
from geosave_engine.workflow.spec import ModelSpec, Ref
from geosave_engine.geodata.transform.tiling import Tiles
from geosave_engine.workflow.processing import Processor
spec = ModelSpec(schema_version=2, sources={}, preprocessing={'result': {'call': 'builtins.dict'}})
assert Processor(stage="preprocessing", spec=spec)({})['result'] == {}
assert not any(name == root or name.startswith(root + '.') for name in sys.modules
               for root in ('prefect', 'lightning', 'pytorch_lightning', 'geosave_engine.workflow.flows'))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage", ["inference", "sources", "unknown"])
def test_only_implemented_processing_stages_can_execute(stage):
    with pytest.raises(ValueError, match="stage"):
        Processor(spec(), stage=stage)


def test_postprocessing_accumulates_batches_and_drains_completed_rasters(raw):
    optical = raw["optical"]
    tiles = Tiles([optical], tile_shape=[2, 2])
    merger = tiles.merger()
    configuration = spec(
        postprocessing={
            "added": {
                "call": Ref("merger.add"),
                "kwargs": {"results": Ref("predictions")},
            },
            "completed": {"call": Ref("merger.merge")},
        }
    )
    collect = Processor(configuration, stage="postprocessing")
    for index in range(len(tiles)):
        collected = collect(
            {
                "merger": merger,
                "predictions": {index: np.full((2, 2), 0.5, dtype="float32")},
            }
        )
        assert collected["added"] is None
        if index < len(tiles) - 1:
            assert collected["completed"] == {}
    merged = collected["completed"][0]
    assert merged.gs.geobox == optical.gs.geobox
    np.testing.assert_allclose(merged, 0.5)
    assert merger.merge() == {}
