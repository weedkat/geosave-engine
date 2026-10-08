from dask.callbacks import Callback
import pytest
import xarray as xr

from geosave_engine.model.spec import ModelSpec, Ref, StageSpec

_events: list[int] = []


def record(*, value: int) -> int:
    _events.append(value)
    return value


def stage(**calls: object) -> StageSpec:
    return StageSpec.model_validate(calls)


def test_stage_runs_declarations_in_order() -> None:
    _events.clear()
    declarations = stage(
        first={"call": f"{__name__}.record", "kwargs": {"value": 1}},
        second={"call": f"{__name__}.record", "kwargs": {"value": 2}},
    )

    assert declarations.run({}) == {"first": 1, "second": 2}
    assert _events == [1, 2]


def test_stage_returns_only_results_without_mutating_inputs() -> None:
    supplied = {"value": 3, "unused": object()}
    declarations = stage(
        squared={
            "call": "builtins.pow",
            "kwargs": {"base": Ref("value"), "exp": 2},
        },
        ratio={"call": Ref("squared.as_integer_ratio")},
    )

    assert declarations.run(supplied) == {"squared": 9, "ratio": (9, 1)}
    assert supplied["value"] == 3


def test_stage_validates_all_references_before_calling() -> None:
    _events.clear()
    declarations = stage(
        first={"call": f"{__name__}.record", "kwargs": {"value": 1}},
        second={"call": Ref("missing")},
    )

    with pytest.raises(ValueError, match="missing"):
        declarations.run({})

    assert _events == []


def test_stage_rejects_forward_references_before_calling() -> None:
    _events.clear()
    declarations = stage(
        first={"call": Ref("second")},
        second={"call": f"{__name__}.record", "kwargs": {"value": 2}},
    )

    with pytest.raises(ValueError, match="Forward.*second"):
        declarations.run({})

    assert _events == []


def test_stage_allows_explicit_rebinding() -> None:
    declarations = stage(
        value={
            "call": "builtins.pow",
            "kwargs": {"base": Ref("value"), "exp": 2},
        }
    )

    assert declarations.run({"value": 3}) == {"value": 9}


def test_model_spec_rejects_undeclared_preprocessing_roots() -> None:
    with pytest.raises(ValueError, match="declared rasters.*scale.*value"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {},
                "preprocessing": {
                    "scaled": {
                        "call": Ref("scale"),
                        "kwargs": {"value": Ref("value")},
                    }
                },
            }
        )


def test_preprocess_validates_rasters_before_calls() -> None:
    _events.clear()
    model = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["red"]}},
            "preprocessing": {
                "result": {
                    "call": f"{__name__}.record",
                    "kwargs": {"value": 1, "image": Ref("optical")},
                }
            },
        }
    )

    with pytest.raises(KeyError, match="red"):
        model.preprocess({"optical": xr.Dataset()})

    assert _events == []


@pytest.mark.parametrize(
    ("selector", "names"),
    [
        ({"variables": ["nir", "red"]}, ["nir", "red"]),
        ({"channels": 2}, ["red", "nir"]),
    ],
)
def test_preprocess_selects_each_supplied_raster_without_computing(
    raw, selector: dict[str, object], names: list[str]
) -> None:
    source = raw["optical"]
    model = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {
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
        result = model.preprocess({"optical": source})

    selected = result["result"]["image"]
    assert computed == []
    assert set(result) == {"optical", "result"}
    assert result["optical"] is selected
    assert list(selected.data_vars) == names
    assert selected.red.data is source.red.data
    assert list(source.data_vars) == ["red", "nir", "unused"]


def test_preprocess_selects_a_raster_only_a_model_input_reads(raw) -> None:
    model = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["nir"]}},
            "inputs": {"image": Ref("optical")},
        }
    )

    result = model.preprocess({"optical": raw["optical"]})

    assert list(result["optical"].data_vars) == ["nir"]


def test_preprocess_checks_a_supplied_raster_no_call_reads(raw) -> None:
    model = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {
                "optical": {"variables": ["red"]},
                "elevation": {"variables": ["dem"]},
            },
            "preprocessing": {
                "image": {"call": "builtins.dict", "kwargs": {"data": Ref("optical")}}
            },
        }
    )

    with pytest.raises(KeyError, match="dem") as caught:
        model.preprocess({"optical": raw["optical"], "elevation": raw["optical"]})

    assert caught.value.__notes__ == ["While validating raster 'elevation'"]


def test_preprocessing_hands_features_a_dataset_and_literal_selectors() -> None:
    import dask.array as da

    from tests.geodata.features.conftest import sentinel

    scene = sentinel({"time": 1, "y": 16, "x": 16})
    model = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {
                "optical": {"variables": ["B10"], "coordinates": ["sun_azimuth"]}
            },
            "preprocessing": {
                "cloud": {
                    "call": "geosave_engine.geodata.features.cirrus_cloud_mask",
                    "kwargs": {"scene": Ref("optical"), "b10": "B10"},
                },
                "cloud_scene": {
                    "call": Ref("optical.assign"),
                    "kwargs": {"cloud": Ref("cloud")},
                },
                "shadow": {
                    "call": "geosave_engine.geodata.features.shadow_mask",
                    "kwargs": {
                        "scene": Ref("cloud_scene"),
                        "cloud": "cloud",
                        "sun_azimuth": "sun_azimuth",
                        "shadow_distance_m": 30,
                    },
                },
                "image": {
                    "call": Ref("optical.assign"),
                    "kwargs": {"cloud": Ref("cloud"), "shadow": Ref("shadow")},
                },
            },
            "inputs": {"image": Ref("image")},
        }
    )

    image = model.preprocess({"optical": scene})["image"]

    assert list(image.data_vars) == ["B10", "cloud", "shadow"]
    assert isinstance(image.shadow.data, da.Array)
    assert image.shadow.dtype == bool
    assert image.gs.geobox == scene.gs.geobox

    with pytest.raises(KeyError, match="sun_azimuth"):
        model.preprocess({"optical": scene.drop_vars("sun_azimuth")})
