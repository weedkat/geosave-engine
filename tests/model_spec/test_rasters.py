from __future__ import annotations

import dask.array as da
from dask.callbacks import Callback
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
import yaml
import xarray as xr
from pydantic import ValidationError

from geosave_engine.geodata.core.raster import raster
from geosave_engine.model_spec import RasterRequirement


def test_persisted_raster_requirements_need_no_acquisition_identity():
    requirement = RasterRequirement(variables=("nir", "red"))
    assert requirement.stac is None
    assert list(requirement.select_raster(packed())) == ["nir", "red"]


def packed() -> xr.Dataset:
    return xr.Dataset(
        {
            "red": (
                ("y", "x"),
                da.ones((2, 3), chunks=(1, 3)),
                {"scale_factor": 0.01, "_FillValue": 0},
            ),
            "nir": (
                ("y", "x"),
                da.ones((2, 3), chunks=(1, 3)),
                {"scale_factor": 0.02, "_FillValue": 0},
            ),
            "unused": (("y", "x"), da.ones((2, 3), chunks=(1, 3))),
        },
        coords={"y": [0, 1], "x": [0, 1, 2]},
        attrs={"provider": "example"},
    )


@pytest.mark.parametrize(
    "selector",
    [{}, {"variables": ["red"], "channels": 1}, {"channels": 0}, {"channels": -1}],
)
def test_exactly_one_nonempty_raster_selector_is_required(selector):
    with pytest.raises(ValidationError):
        RasterRequirement.model_validate(selector)


def test_positional_metadata_accepts_only_wildcard_variable_names():
    with pytest.raises(ValidationError, match="Metadata"):
        RasterRequirement.model_validate(
            {"channels": 2, "attrs": {"data_vars": {"red": {}}}}
        )
    RasterRequirement.model_validate(
        {"channels": 2, "attrs": {"data_vars": {"*": {}}}}
    )


@pytest.mark.parametrize(
    ("selector", "names"),
    [
        ({"variables": ["nir", "red"]}, ["nir", "red"]),
        ({"channels": 2}, ["red", "nir"]),
    ],
)
def test_selection_preserves_order_arrays_and_packing_without_computation(
    selector, names
):
    source = packed()
    requirement = RasterRequirement.model_validate(
        {
            **selector,
            "dims": ["y", "x"],
            "dtypes": ["float64"],
            "attrs": {
                "data_vars": {
                    "*": {"models": {"packing": {"required": ["scale_factor"]}}}
                }
            },
        }
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        selected = requirement.select_raster(source)
        assert requirement.validate_raster(source) is None
    assert tasks == []
    assert list(selected.data_vars) == names
    for name in names:
        assert selected[name].data is source[name].data
        assert selected[name].attrs == source[name].attrs
    assert list(source.data_vars) == ["red", "nir", "unused"]


def test_positional_band_selection_preserves_positions_and_lazy_graph():
    data = da.from_array(np.arange(18), chunks="auto").reshape((3, 2, 3))
    source = xr.Dataset(
        {"image": (("band", "y", "x"), data, {"scale_factor": 0.01})},
        coords={"band": [8, 4, 2]},
    )
    requirement = RasterRequirement(channels=2, dims=("band", "y", "x"))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        selected = requirement.select_raster(source)
        assert requirement.validate_raster(source) is None
    assert tasks == []
    assert list(selected.data_vars) == ["image"]
    assert selected.band.values.tolist() == [8, 4]
    assert selected.image.data.name == data[:2].name
    assert set(data.dask) <= set(selected.image.data.dask)
    assert selected.image.attrs == source.image.attrs
    assert source.sizes["band"] == 3


@pytest.mark.parametrize(
    "layout", ["ordinary", "band", "mixed", "multiple_band", "empty"]
)
def test_positional_selection_rejects_short_or_ambiguous_rasters(layout):
    band = xr.DataArray(da.ones((1, 2, 3), chunks=(1, 2, 3)), dims=("band", "y", "x"))
    source = {
        "ordinary": packed()[["red"]],
        "band": xr.Dataset({"image": band}),
        "mixed": xr.Dataset({"image": band, "red": band.isel(band=0)}),
        "multiple_band": xr.Dataset({"image": band, "other": band}),
        "empty": xr.Dataset(),
    }[layout]
    message = "ambiguous" if layout in ("mixed", "multiple_band") else "channels"
    requirement = RasterRequirement(channels=2)
    with pytest.raises(ValueError, match=message):
        requirement.select_raster(source)


@pytest.mark.parametrize(
    "constraint",
    [
        {"dims": ["time", "y", "x"]},
        {"dtypes": ["uint16"]},
        {
            "attrs": {
                "data_vars": {
                    "*": {"models": {"packing": {"required": ["add_offset"]}}}
                }
            }
        },
    ],
)
def test_positional_selection_validates_selected_variables(constraint):
    requirement = RasterRequirement(channels=2, **constraint)
    with pytest.raises(ValueError):
        requirement.select_raster(packed())


@pytest.mark.parametrize("selector", [{"variables": ["red"]}, {"channels": 1}])
def test_selection_requires_a_dataset(selector):
    with pytest.raises(TypeError, match="Dataset"):
        RasterRequirement(**selector).select_raster(packed().red)


def test_scoped_requirements_validate_selected_bands_without_computing():
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["nir", "red"],
            "dims": ["y", "x"],
            "attrs": {
                "root": {"foreign": {"equals": {"provider": "example"}}},
                "data_vars": {
                    "*": {
                        "models": {
                            "packing": {"required": ["scale_factor"]},
                            "nodata": {"equals": {"fill_value": 0}},
                        }
                    }
                },
            },
        }
    )
    tasks = []
    data = packed()
    with Callback(pretask=lambda *args: tasks.append(args)):
        requirement.validate_raster(data)
    assert tasks == []
    assert isinstance(data.red.data, da.Array)


def test_an_optional_attrs_field_must_actually_be_present():
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["red"],
            "attrs": {
                "data_vars": {
                    "red": {
                        "models": {
                            "packing": {"required": ["add_offset"]},
                        }
                    }
                }
            },
        }
    )
    with pytest.raises(ValueError, match="red.*packing.*add_offset"):
        requirement.validate_raster(packed())


@pytest.mark.parametrize(
    "models",
    [
        {"not_registered": {"required": ["value"]}},
        {"packing": {"required": ["scale_fator"]}},
        {"packing": {"equals": {"scale_factor": "not-a-number"}}},
        {"packing": {"one_of": {"scale_factor": []}}},
    ],
)
def test_invalid_attrs_requirements_fail_at_spec_creation(models):
    with pytest.raises(ValidationError):
        RasterRequirement.model_validate(
            {
                "variables": ["red"],
                "attrs": {"data_vars": {"red": {"models": models}}},
            }
        )


def test_foreign_requirements_cannot_bypass_registered_models():
    with pytest.raises(ValidationError, match="registered"):
        RasterRequirement.model_validate(
            {
                "variables": ["red"],
                "attrs": {
                    "data_vars": {
                        "red": {
                            "foreign": {"required": ["scale_factor"]},
                        }
                    }
                },
            }
        )


def test_required_coordinates_are_present_without_computing():
    data = packed().drop_indexes("x").drop_vars("x")
    requirement = RasterRequirement(variables=("red",), coordinates=("x",))
    tasks = []

    with Callback(pretask=lambda *args: tasks.append(args)):
        with pytest.raises(ValueError, match="coordinates.*x"):
            requirement.validate_raster(data)

    assert tasks == []


def test_extra_coordinates_are_allowed_and_retained():
    requirement = RasterRequirement(
        variables=("red",), coordinates=("x",), dims=("y", "x")
    )

    selected = requirement.select_raster(packed())

    assert tuple(selected.coords) == ("y", "x")


def test_coordinate_attrs_still_apply():
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["red"],
            "attrs": {
                "coords": {
                    "x": {
                        "models": {
                            "coordinate": {"one_of": {"axis": ["X"]}},
                        }
                    }
                }
            },
        }
    )
    data = packed()
    data.x.attrs["axis"] = "Y"
    with pytest.raises(ValueError, match="x.*axis"):
        requirement.validate_raster(data)
    data.x.attrs["axis"] = "X"
    requirement.validate_raster(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"variables": []},
        {"variables": ["red", "red"]},
        {"dims": ["y", "y"]},
        {"attrs": {"data_vars": {"missing": {}}}},
    ],
)
def test_ambiguous_or_unreachable_requirements_are_rejected(changes):
    with pytest.raises(ValidationError):
        RasterRequirement.model_validate({"variables": ["red"], **changes})


def test_missing_variables_dimensions_and_dtype_fail_without_conversion():
    data = packed()
    for requirement in (
        RasterRequirement(variables=("missing",)),
        RasterRequirement(variables=("red",), dims=("time", "y", "x")),
        RasterRequirement(variables=("red",), dtypes=("uint16",)),
    ):
        with pytest.raises(ValueError):
            requirement.validate_raster(data)
    assert data.red.dtype == "float64"


@pytest.mark.parametrize("value", [np.array([1.0, 2.0]), (1.0, 2.0)])
@pytest.mark.parametrize(
    "predicate",
    [
        {"equals": {"calibration": [1.0, 2.0]}},
        {"one_of": {"calibration": [[1.0, 2.0]]}},
    ],
)
def test_foreign_array_metadata_matches_sequences(value, predicate):
    requirement = RasterRequirement.model_validate(
        {"variables": ["red"], "attrs": {"root": {"foreign": predicate}}}
    )
    data = packed()
    data.attrs["calibration"] = value
    requirement.validate_raster(data)
    data.attrs["calibration"] = np.array([1.0, 3.0])
    with pytest.raises(ValueError, match="calibration"):
        requirement.validate_raster(data)


@pytest.mark.parametrize(
    "predicate",
    [
        {"equals": {"TIFFTAG_DATETIME": "2026-09-23T12:00:00"}},
        {"one_of": {"TIFFTAG_DATETIME": ["2026-09-23T12:00:00"]}},
    ],
)
def test_registered_datetime_uses_the_same_typed_representation_on_both_sides(
    predicate,
):
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["red"],
            "attrs": {"root": {"models": {"geotiff": predicate}}},
        }
    )
    data = packed()
    data.attrs["TIFFTAG_DATETIME"] = "2026:09:23 12:00:00"
    requirement.validate_raster(data)
    data.attrs["TIFFTAG_DATETIME"] = "2026:09:23 12:00:01"
    with pytest.raises(ValueError, match="TIFFTAG_DATETIME"):
        requirement.validate_raster(data)


def test_crs_requirement_applies_to_selected_variables_not_an_unused_raster():
    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32633", resolution=10)
    data = raster({"unused": np.ones(tuple(box.shape))}, box).assign(offset=1.0)
    requirement = RasterRequirement(variables=("offset",), require_crs=True)
    with pytest.raises(ValueError, match="CRS"):
        requirement.validate_raster(data)


def test_partial_predicates_use_attrs_field_parsing_without_constructing_whole_models():
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["red"],
            "attrs": {
                "data_vars": {
                "red": {
                    "models": {
                        "legend": {"equals": {"flag_values": "[0, 1]"}},
                    }
                }
                }
            },
        }
    )
    data = packed()
    data.red.attrs.update(flag_values=[0, 1], flag_meanings="background water")
    requirement.validate_raster(data)


def test_registered_predicate_consistency_uses_parsed_field_values():
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["red"],
            "attrs": {
                "data_vars": {
                "red": {
                    "models": {
                        "packing": {
                            "equals": {"scale_factor": "0.01"},
                            "one_of": {"scale_factor": [0.01]},
                        },
                    }
                }
                }
            },
        }
    )
    requirement.validate_raster(packed())
    restored = RasterRequirement.model_validate(
        yaml.safe_load(yaml.safe_dump(requirement.model_dump()))
    )
    restored.validate_raster(packed())


@pytest.mark.parametrize(
    "predicate",
    [
        {"legend": {"equals": {"flag_values": "null"}}},
        {"gdal": {"equals": {"variable_name": ""}}},
        {"packing": {"equals": {"scale_factor": 1}, "one_of": {"scale_factor": [2]}}},
    ],
)
def test_invalid_parsed_predicates_fail_at_spec_creation(predicate):
    with pytest.raises(ValidationError):
        RasterRequirement.model_validate(
            {
                "variables": ["red"],
                "attrs": {"data_vars": {"red": {"models": predicate}}},
            }
        )


@pytest.mark.parametrize("resolution", [10, (10, 10)])
def test_resolution_is_checked_without_computing_pixels(raw, resolution):
    from dask.callbacks import Callback

    tasks = []
    requirement = RasterRequirement(variables=("nir",), resolution=resolution)
    with Callback(pretask=lambda *args: tasks.append(args)):
        requirement.validate_raster(raw["optical"])
    assert tasks == []
    with pytest.raises(ValueError, match="resolution"):
        RasterRequirement(variables=("nir",), resolution=20).validate_raster(
            raw["optical"]
        )


@pytest.mark.parametrize("resolution", [0, -1, (10, 0), (10, -1), float("inf")])
def test_resolution_must_be_positive_and_finite(resolution):
    with pytest.raises(ValueError):
        RasterRequirement(variables=("red",), resolution=resolution)
