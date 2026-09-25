from __future__ import annotations

import pytest
from pydantic import ValidationError
import yaml

from geosave_engine.workflow.spec import ModelSpec, OperationSpec, TimeWindowSpec

CALL = "geosave_engine.geodata.transform.nodata.to_nan"


def config():
    return {
        "sources": {"optical-l2.v1": {"variables": ["nir", "red"]}},
        "preprocessing": {
            "features": {"raster": "optical-l2.v1", "operations": [{"call": CALL}]}
        },
        "inference": {
            "inputs": {"image": {"raster": "features"}},
            "tiling": {"raster": "features", "tile_shape": [4, 4]},
        },
        "postprocessing": {"classes": ["other", "water"]},
    }


@pytest.mark.parametrize("name", ["model", "custom.yaml", "custom.yml"])
def test_yaml_round_trip_preserves_model_files(tmp_path, name):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    spec = ModelSpec.model_validate(config())
    path = spec.save(tmp_path / name)
    assert path.suffix in (".yaml", ".yml")
    assert ModelSpec.load(path) == spec
    assert ModelSpec.load(tmp_path / name) == spec
    assert yaml.safe_load(path.read_text())["schema_version"] == 1
    assert weights.read_bytes() == b"weights"


@pytest.mark.parametrize(
    "payload", ["sources: {}\nsources: {}", "sources:\n  optical: {}\n  optical: {}"]
)
def test_duplicate_yaml_keys_are_rejected(tmp_path, payload):
    path = tmp_path / "model_spec.yaml"
    path.write_text(payload)
    with pytest.raises(ValueError, match="Duplicate"):
        ModelSpec.load(path)


def test_loading_does_not_execute_operations(tmp_path, monkeypatch):
    from geosave_engine.geodata.transform import nodata

    def fail(data):
        pytest.fail("operation ran while loading")

    monkeypatch.setattr(nodata, "to_nan", fail)
    spec = ModelSpec.model_validate(config())
    assert ModelSpec.load(spec.save(tmp_path)) == spec


@pytest.mark.parametrize(
    "operation",
    [
        {"call": "missing_module.transform"},
        {"call": "geosave_engine.geodata.transform.nodata.np"},
        {"call": CALL, "kwargs": {"typo": 1}},
        {"call": "geosave_engine.geodata.transform.nodata.mask"},
        {"call": CALL, "kwargs": {"data": 1}},
        {"call": CALL, "kwargs": {"x": float("nan")}},
    ],
)
def test_invalid_callable_and_arguments_fail(operation):
    with pytest.raises(ValueError):
        OperationSpec.model_validate(operation)


@pytest.mark.parametrize(
    "recipes,match",
    [
        ({"optical-l2.v1": {"raster": "optical-l2.v1"}}, "collid"),
        ({"features": {"raster": "absent"}}, "unknown"),
        ({"features": {"raster": "other"}, "other": {"raster": "features"}}, "cycl"),
        (
            {
                "features": {
                    "raster": "optical-l2.v1",
                    "operations": [
                        {"call": "xarray.merge", "inputs": {"compat": "absent"}}
                    ],
                }
            },
            "unknown",
        ),
    ],
)
def test_invalid_recipe_graphs_fail(recipes, match):
    value = config()
    value["preprocessing"] = recipes
    with pytest.raises(ValueError, match=match):
        ModelSpec.model_validate(value)


def test_forward_references_work():
    value = config()
    value["preprocessing"] = {
        "features": {"raster": "middle"},
        "middle": {"raster": "optical-l2.v1"},
    }
    assert ModelSpec.model_validate(value).preparation_order() == ("middle", "features")


@pytest.mark.parametrize(
    "change", [{"schema_version": 2}, {"unexpected": True}, {"prediction": {}}]
)
def test_unknown_schema_and_keys_are_rejected(change):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({**config(), **change})


def test_invalid_bindings_and_normalization_fail():
    value = config()
    value["inference"]["inputs"]["image"] = {"raster": "absent"}
    with pytest.raises(ValueError, match="unknown"):
        ModelSpec.model_validate(value)
    value["inference"]["inputs"]["image"] = {
        "raster": "optical-l2.v1",
        "normalize": {"mean": [0], "std": [1]},
    }
    with pytest.raises(ValueError, match="channels"):
        ModelSpec.model_validate(value)


@pytest.mark.parametrize("name", ["a/b", ".", "..", ""])
def test_raster_names_are_flat(name):
    value = config()
    value["sources"] = {name: {"variables": ["red"]}}
    with pytest.raises(ValueError):
        ModelSpec.model_validate(value)


def test_model_argument_names_are_identifiers():
    value = config()
    value["inference"]["inputs"] = {"not-valid": {"raster": "features"}}
    with pytest.raises(ValueError):
        ModelSpec.model_validate(value)


@pytest.mark.parametrize(
    "values",
    [
        {"size": 0, "tolerance": "1D"},
        {"size": 2, "tolerance": "0D"},
        {"size": 2, "tolerance": "-1D"},
        {"size": 2, "tolerance": "bad"},
        {"size": 2, "tolerance": "1D", "stride": 0},
    ],
)
def test_invalid_temporal_settings_fail(values):
    with pytest.raises(ValueError):
        TimeWindowSpec(**values)


def test_mutation_is_revalidated_before_save(tmp_path):
    spec = ModelSpec.model_validate(config())
    path = spec.save(tmp_path)
    original = path.read_text()
    spec.sources.clear()
    with pytest.raises(ValidationError):
        spec.save(tmp_path)
    assert path.read_text() == original


def test_uri_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="local"):
        ModelSpec.model_validate(config()).save("s3://bucket/model")
    assert not list(tmp_path.iterdir())


async def async_operation(data):
    return data


def generator_operation(data):
    yield data


class AsyncOperation:
    async def __call__(self, data):
        return data


class GeneratorOperation:
    def __call__(self, data):
        yield data


async_instance = AsyncOperation()
generator_instance = GeneratorOperation()


@pytest.mark.parametrize(
    "path",
    [
        f"{__name__}.async_operation",
        f"{__name__}.generator_operation",
        f"{__name__}.AsyncOperation",
        f"{__name__}.async_instance",
        f"{__name__}.generator_instance",
        "geosave_engine.geodata.transform.nodata:to_nan",
    ],
)
def test_unsupported_callable_types_and_path_forms_fail(path):
    with pytest.raises(ValueError):
        OperationSpec(call=path)


def test_prefect_tasks_resolve_to_plain_function(monkeypatch):
    from prefect import task
    from geosave_engine.geodata.transform import nodata

    original = nodata.to_nan
    monkeypatch.setattr(nodata, "to_nan", task(original))
    assert OperationSpec(call=CALL).resolve() is original


def test_postprocessing_cannot_reference_prepared_rasters():
    value = config()
    value["postprocessing"] = {
        "call": "xarray.merge",
        "inputs": {"compat": "features"},
    }
    with pytest.raises(ValueError, match="Postprocessing"):
        ModelSpec.model_validate(value)


def test_python_yaml_tags_are_not_executed(tmp_path):
    path = tmp_path / "model_spec.yaml"
    path.write_text("!!python/object/apply:builtins.print [should-not-execute]")
    with pytest.raises(yaml.constructor.ConstructorError):
        ModelSpec.load(path)


@pytest.mark.parametrize(
    "tiling",
    [
        {"tile_shape": [0, 4]},
        {"overlap": -1},
        {"overlap": 4},
        {"window": "hann"},
    ],
)
def test_invalid_spatial_tiling_fails(tiling):
    value = config()
    value["inference"]["tiling"].update(tiling)
    with pytest.raises(ValueError):
        ModelSpec.model_validate(value)


@pytest.mark.parametrize(
    "binding",
    [
        {"normalize": {"mean": [0, 0], "std": [1, 0]}},
        {"normalize": {"mean": [0], "std": [1, 1]}},
        {"variables": ["nir", "nir"]},
        {"dtype": "int64"},
        {"layout": "HWC"},
    ],
)
def test_invalid_tensor_encoding_fails(binding):
    value = config()
    value["inference"]["inputs"]["image"].update(binding)
    with pytest.raises(ValueError):
        ModelSpec.model_validate(value)


@pytest.mark.parametrize(
    "interpretation",
    [
        {"classes": ["water", "water"]},
        {"thresholds": [0.5]},
        {"ignore_index": 1},
    ],
)
def test_invalid_segmentation_settings_fail(interpretation):
    value = config()
    value["postprocessing"].update(interpretation)
    with pytest.raises(ValueError):
        ModelSpec.model_validate(value)


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json"])
def test_existing_non_yaml_files_are_rejected_without_modification(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(b"model artifact bytes")
    spec = ModelSpec.model_validate(config())
    with pytest.raises(ValueError, match="YAML"):
        spec.save(path)
    assert path.read_bytes() == b"model artifact bytes"
    with pytest.raises(ValueError, match="YAML"):
        ModelSpec.load(path)
    assert path.read_bytes() == b"model artifact bytes"


@pytest.mark.parametrize("filename", ["model.safetensors", "model_spec.json"])
def test_missing_non_yaml_files_are_rejected_without_creating_directories(
    tmp_path, filename
):
    path = tmp_path / filename
    spec = ModelSpec.model_validate(config())
    with pytest.raises(ValueError, match="YAML"):
        spec.save(path)
    assert not path.exists()
    with pytest.raises(ValueError, match="YAML"):
        ModelSpec.load(path)


@pytest.mark.parametrize("name", ["model.safetensors", "directory.yaml"])
def test_existing_artifact_directories_accept_any_suffix(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    spec = ModelSpec.model_validate(config())
    assert spec.save(directory) == directory / ModelSpec.filename
    assert ModelSpec.load(directory) == spec


@pytest.mark.parametrize(
    "classes",
    [
        ["dry land", "water"],
        ["background", "water\tclass"],
        ["background", " water"],
    ],
)
def test_segmentation_names_follow_native_legend_validation(classes):
    value = config()
    value["postprocessing"]["classes"] = classes
    with pytest.raises(ValueError, match="whitespace"):
        ModelSpec.model_validate(value)


def test_segmentation_names_preserve_native_legend_tokens(tmp_path):
    from geosave_engine.geodata.attrs import Legend

    value = config()
    value["postprocessing"]["classes"] = ["dry_land", "water-body"]
    spec = ModelSpec.model_validate(value)
    restored = ModelSpec.load(spec.save(tmp_path))
    legend = Legend(class_map=dict(enumerate(restored.postprocessing.classes)))
    assert legend.class_map == {0: "dry_land", 1: "water-body"}


@pytest.mark.parametrize(
    "operation",
    [
        {},
        {"call": CALL, "method": "gs.to_nan"},
        {"method": "gs.to_nan()"},
    ],
)
def test_operation_requires_one_plain_callable_selector(operation):
    with pytest.raises(ValueError):
        OperationSpec.model_validate(operation)


def test_native_method_recipe_round_trips_without_executing(tmp_path, monkeypatch):
    from geosave_engine.geodata.core.raster import GeoRaster

    def fail(self):
        pytest.fail("method executed while loading")

    monkeypatch.setattr(GeoRaster, "to_nan", fail)
    value = config()
    value["preprocessing"]["features"] = {
        "raster": "optical-l2.v1",
        "variables": ["red", "nir"],
        "operations": [{"method": "gs.to_nan"}],
    }
    spec = ModelSpec.model_validate(value)
    assert ModelSpec.load(spec.save(tmp_path)) == spec
    assert spec.preprocessing["features"].variables == ("red", "nir")


@pytest.mark.parametrize("variables", [[], ["red", "red"]])
def test_recipe_selection_must_be_nonempty_and_unique(variables):
    value = config()
    value["preprocessing"]["features"]["variables"] = variables
    with pytest.raises(ValueError):
        ModelSpec.model_validate(value)


def test_method_apply_supports_native_dataarray_postprocessing():
    import xarray as xr

    data = xr.DataArray([1.0, 2.0], dims="band")
    operation = OperationSpec(method="to_dataset", kwargs={"name": "logits"})
    output = operation.apply(data)
    assert output.logits.identical(data.rename("logits"))
