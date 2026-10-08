"""Each attrs mapping is parsed against the models of the one scope it lives in."""

import pytest
import xarray as xr

from geosave_engine.geodata.attrs import (
    ACDD,
    AttrsNamespace,
    CFCoordinate,
    CFVariable,
    Nodata,
    Packing,
    create_header,
)


def test_a_variable_with_units_carries_only_variable_semantics() -> None:
    namespace = AttrsNamespace.from_attrs({"units": "1"}, "variable")

    assert namespace.get(CFVariable) == CFVariable(units="1")
    assert namespace.get(CFCoordinate) is None


def test_a_coordinate_with_units_carries_only_coordinate_semantics() -> None:
    header = create_header(xr.Dataset(coords={"x": ("x", [0], {"units": "metre"})}))

    assert header.coords["x"].get(CFCoordinate) == CFCoordinate(units="metre")
    assert header.coords["x"].get(CFVariable) is None


def test_nodata_on_a_dataset_root_is_foreign() -> None:
    header = create_header(xr.Dataset(attrs={"nodata": 0}))

    assert header.root.get(Nodata) is None
    assert header.root.foreign == {"nodata": 0}


def test_fill_value_on_a_coordinate_is_foreign() -> None:
    header = create_header(xr.Dataset(coords={"x": ("x", [0.0], {"_FillValue": 0.0})}))

    assert header.coords["x"].foreign == {"_FillValue": 0.0}


def test_a_dataarray_root_is_one_variable() -> None:
    header = create_header(xr.DataArray([1], dims="x", attrs={"nodata": 0}))

    assert header.root.scope == "variable"
    assert header.root.get(Nodata) == Nodata(fill_value=0)


def test_a_namespace_refuses_models_of_different_scopes() -> None:
    with pytest.raises(ValueError, match="different scopes"):
        AttrsNamespace(models={Nodata: Nodata(fill_value=0), ACDD: ACDD(title="x")})


def test_a_namespace_refuses_a_foreign_key_its_models_write() -> None:
    with pytest.raises(ValueError, match="collide"):
        AttrsNamespace(models={Nodata: Nodata(fill_value=0)}, foreign={"nodata": 0})


def test_namespaces_of_different_scopes_do_not_merge() -> None:
    dataset = AttrsNamespace(models={ACDD: ACDD(title="x")})
    variable = AttrsNamespace(models={CFVariable: CFVariable(long_name="Red")})

    with pytest.raises(ValueError, match="different scopes"):
        AttrsNamespace.merge([dataset, variable])


def test_grouping_extracts_owned_keys_without_changing_the_source():
    source = {
        "units": "1",
        "nodata": "0",
        "_FillValue": 0,
        "scale_factor": 0.1,
        "vendor": {"site": 7},
    }

    namespace = AttrsNamespace.from_attrs(source, "variable")

    assert set(namespace.models) == {CFVariable, Nodata, Packing}
    assert namespace.get(Nodata).fill_value == 0
    assert namespace.get("nodata") is namespace.get(Nodata)
    assert namespace.foreign == {"vendor": {"site": 7}}
    assert source == {
        "units": "1",
        "nodata": "0",
        "_FillValue": 0,
        "scale_factor": 0.1,
        "vendor": {"site": 7},
    }


def test_invalid_owned_attrs_do_not_become_foreign_metadata():
    with pytest.raises(ValueError):
        AttrsNamespace.from_attrs(
            {"nodata": "invalid", "vendor": "example"}, "variable"
        )


def test_storage_keys_are_distinct_from_logical_field_names():
    assert Nodata.keys_for("fill_value") == ("_FillValue", "nodata")
    assert Nodata.attr_keys() == ("_FillValue", "nodata")
    assert Packing.keys_for("scale_factor") == ("scale_factor",)
    with pytest.raises(KeyError, match="typo"):
        Nodata.keys_for("typo")


def test_namespace_owns_flat_attrs_lifecycle() -> None:
    first = AttrsNamespace.from_attrs({"title": "source", "provider": "one"}, "dataset")
    second = AttrsNamespace.from_attrs(
        {"title": "source", "provider": "two"}, "dataset"
    )

    merged, dropped = AttrsNamespace.merge([first, second])

    assert merged.to_attrs() == {"title": "source"}
    assert dropped == {"provider"}


@pytest.mark.parametrize("second", [{}, {"units": "K", "scale_factor": 0.2}])
def test_namespace_can_drop_conflicting_required_fields(second) -> None:
    first = AttrsNamespace.from_attrs(
        {"units": "m", "scale_factor": 0.1, "long_name": "Height"}, "variable"
    )
    other = AttrsNamespace.from_attrs({**second, "long_name": "Height"}, "variable")

    merged, dropped = AttrsNamespace.merge([first, other], conflicts="drop")

    assert merged.to_attrs() == {"long_name": "Height"}
    assert dropped == {"units", "scale_factor"}
    with pytest.raises(ValueError, match="must agree"):
        AttrsNamespace.merge([first, other])
