from __future__ import annotations

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from geosave_engine.workflow.preprocessing import preprocess
from geosave_engine.workflow.spec import ModelSpec


def scale(data, *, factor=2, other=None):
    result = data.copy()
    result["nir"] = data.nir * factor + (other.nir if other is not None else 0)
    return result


def mutate(data, *, other):
    data.nir.values[:] = 99
    data.nir.attrs["nested"]["value"] = 99
    other.nir.values[:] = 88
    other.attrs["changed"] = True
    return data


def wrong_result(data):
    return data.nir


def make_spec(recipes):
    return ModelSpec(
        sources={"optical": {"variables": ["nir", "red"], "require_crs": True}},
        preprocessing=recipes,
        inference={
            "inputs": {"image": {"raster": "features"}},
            "tiling": {"raster": "features", "tile_shape": [4, 4]},
        },
    )


def test_forward_branching_preserves_sources_and_laziness(raw):
    spec = make_spec(
        {
            "features": {
                "raster": "middle",
                "operations": [
                    {
                        "call": f"{__name__}.scale",
                        "kwargs": {"factor": 3},
                        "inputs": {"other": "optical"},
                    }
                ],
            },
            "middle": {
                "raster": "optical",
                "operations": [{"call": f"{__name__}.scale"}],
            },
        }
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        prepared = preprocess(xr.DataTree.from_dict(raw), spec=spec)
    assert tasks == []
    assert isinstance(prepared, xr.DataTree)
    assert set(prepared.children) == {"optical", "middle", "features"}
    result = prepared["features"].to_dataset()
    assert list(result.data_vars) == ["nir", "red"]
    assert isinstance(result.nir.data, da.Array)
    assert result.gs.geobox == raw["optical"].gs.geobox
    np.testing.assert_allclose(result.nir.compute(), 42)
    np.testing.assert_allclose(prepared["optical"].nir.compute(), 6)
    assert list(raw["optical"].data_vars) == ["red", "nir", "unused"]


def test_mutating_operations_cannot_change_sources_or_other_branches(raw):
    raw = {"optical": raw["optical"].compute()}
    raw["optical"].nir.attrs["nested"] = {"value": 1}
    spec = make_spec(
        {
            "features": {
                "raster": "optical",
                "operations": [
                    {"call": f"{__name__}.mutate", "inputs": {"other": "optical"}}
                ],
            },
            "sibling": {"raster": "optical"},
        }
    )
    prepared = preprocess(raw, spec=spec)
    for data in (
        raw["optical"],
        prepared["optical"].to_dataset(),
        prepared["sibling"].to_dataset(),
    ):
        np.testing.assert_allclose(data.nir, 6)
        assert data.nir.attrs["nested"] == {"value": 1}
        assert "changed" not in data.attrs


def test_operations_must_return_datasets(raw):
    spec = make_spec(
        {
            "features": {
                "raster": "optical",
                "operations": [{"call": f"{__name__}.wrong_result"}],
            }
        }
    )
    with pytest.raises(TypeError, match="Dataset"):
        preprocess(raw, spec=spec)


def test_source_validation_runs_before_operations(raw):
    spec = make_spec(
        {
            "features": {
                "raster": "optical",
                "operations": [{"call": f"{__name__}.wrong_result"}],
            }
        }
    )
    with pytest.raises(ValueError, match="nir"):
        preprocess({"optical": raw["optical"].drop_vars("nir")}, spec=spec)


def test_mutated_calls_revalidate_before_pixels(raw):
    spec = make_spec(
        {
            "features": {
                "raster": "optical",
                "operations": [{"call": f"{__name__}.scale"}],
            }
        }
    )
    spec.preprocessing["features"].operations[0].kwargs["typo"] = 1
    tasks = []
    with (
        Callback(pretask=lambda *args: tasks.append(args)),
        pytest.raises(ValueError, match="typo"),
    ):
        preprocess(raw, spec=spec)
    assert tasks == []


@pytest.mark.parametrize("name", ["x", "y", "spatial_ref"])
@pytest.mark.parametrize("role", ["source", "recipe"])
def test_shared_root_coordinate_names_fail_before_operations(raw, name, role):
    source = name if role == "source" else "optical"
    output = name if role == "recipe" else "features"
    spec = ModelSpec(
        sources={source: {"variables": ["nir", "red"], "require_crs": True}},
        preprocessing={
            output: {
                "raster": source,
                "operations": [{"call": f"{__name__}.wrong_result"}],
            }
        },
        inference={
            "inputs": {"image": {"raster": output}},
            "tiling": {"raster": output, "tile_shape": [4, 4]},
        },
    )
    tasks = []
    with (
        Callback(pretask=lambda *args: tasks.append(args)),
        pytest.raises(ValueError, match="root coordinate.*" + name),
    ):
        preprocess({source: raw["optical"]}, spec=spec)
    assert tasks == []


def test_coordinate_names_are_not_reserved_without_shared_source_grid(raw):
    optical = raw["optical"]
    smaller = optical.isel(x=slice(0, 2), y=slice(0, 2))
    spec = ModelSpec(
        sources={name: {"variables": ["nir", "red"]} for name in ("x", "other")},
        inference={
            "inputs": {"image": {"raster": "x"}},
            "tiling": {"raster": "x", "tile_shape": [4, 4]},
        },
    )
    prepared = preprocess({"x": optical, "other": smaller}, spec=spec)
    assert set(prepared.children) == {"x", "other"}
    assert prepared["x"].to_dataset().gs.geobox == optical.gs.geobox
    assert prepared["other"].to_dataset().gs.geobox == smaller.gs.geobox


def test_native_accessor_operations_preserve_selected_order_and_lazy_sources(raw):
    for variable in raw["optical"].data_vars.values():
        variable.attrs.update(_FillValue=0, scale_factor=0.5)
    spec = make_spec(
        {
            "features": {
                "raster": "optical",
                "variables": ["red", "nir"],
                "operations": [{"method": "gs.to_nan"}, {"method": "gs.unpack"}],
            }
        }
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        prepared = preprocess(raw, spec=spec)
    assert tasks == []
    features = prepared["features"].to_dataset()
    assert list(features.data_vars) == ["red", "nir"]
    np.testing.assert_allclose(features.red.compute(), 1)
    np.testing.assert_allclose(features.nir.compute(), 3)
    assert raw["optical"].red.attrs["scale_factor"] == 0.5
    assert prepared["optical"].red.attrs["scale_factor"] == 0.5


def test_native_dataset_methods_use_the_real_keyword_api(raw):
    spec = make_spec(
        {
            "features": {
                "raster": "optical",
                "variables": ["red"],
                "operations": [
                    {"method": "isel", "kwargs": {"x": [0, 1]}},
                    {"method": "assign_coords", "kwargs": {"sensor": "example"}},
                ],
            }
        }
    )
    prepared = preprocess(raw, spec=spec)
    assert prepared["features"].sizes["x"] == 2
    assert prepared["features"].coords["sensor"].item() == "example"
    assert raw["optical"].sizes["x"] == 4
    assert "sensor" not in raw["optical"].coords


@pytest.mark.parametrize(
    "operation",
    [
        {"method": "gs.missing"},
        {"method": "gs.to_nan", "kwargs": {"typo": 1}},
        {"method": "attrs"},
    ],
)
def test_native_methods_are_preflighted_before_any_operation(raw, operation):
    spec = make_spec(
        {
            "early": {
                "raster": "optical",
                "operations": [{"call": f"{__name__}.wrong_result"}],
            },
            "features": {"raster": "optical", "operations": [operation]},
        }
    )
    tasks = []
    with (
        Callback(pretask=lambda *args: tasks.append(args)),
        pytest.raises(ValueError, match="method|arguments|callable"),
    ):
        preprocess(raw, spec=spec)
    assert tasks == []


def test_missing_source_recipe_variables_fail_before_any_operation(raw):
    spec = make_spec(
        {
            "early": {
                "raster": "optical",
                "operations": [{"call": f"{__name__}.wrong_result"}],
            },
            "features": {"raster": "optical", "variables": ["missing"]},
        }
    )
    with pytest.raises(ValueError, match="features.*missing"):
        preprocess(raw, spec=spec)


def test_native_method_inputs_can_reference_other_rasters(raw):
    spec = make_spec(
        {
            "features": {
                "raster": "optical",
                "variables": ["red"],
                "operations": [
                    {
                        "method": "merge",
                        "kwargs": {"compat": "no_conflicts"},
                        "inputs": {"other": "nir_only"},
                    },
                ],
            },
            "nir_only": {"raster": "optical", "variables": ["nir"]},
        }
    )
    prepared = preprocess(raw, spec=spec)
    assert list(prepared["features"].data_vars) == ["red", "nir"]
