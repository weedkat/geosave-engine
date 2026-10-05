"""Each attrs mapping is parsed against the models of the one scope it lives in."""

import pytest
import xarray as xr

from geosave_engine.geodata.attrs import (
    ACDD,
    AttrsNamespace,
    CFCoordinate,
    CFVariable,
    Nodata,
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
        AttrsNamespace(models={"nodata": Nodata(fill_value=0), "acdd": ACDD(title="x")})


def test_a_namespace_refuses_a_foreign_key_its_models_write() -> None:
    with pytest.raises(ValueError, match="collide"):
        AttrsNamespace(models={"nodata": Nodata(fill_value=0)}, foreign={"nodata": 0})


def test_namespaces_of_different_scopes_do_not_merge() -> None:
    dataset = AttrsNamespace(models={"acdd": ACDD(title="x")})
    variable = AttrsNamespace(models={"cf_variable": CFVariable(long_name="Red")})

    with pytest.raises(ValueError, match="different scopes"):
        AttrsNamespace.merge([dataset, variable])


def test_namespace_owns_flat_attrs_lifecycle() -> None:
    first = AttrsNamespace.from_attrs({"title": "source", "provider": "one"}, "dataset")
    second = AttrsNamespace.from_attrs({"title": "source", "provider": "two"}, "dataset")

    merged, dropped = AttrsNamespace.merge([first, second])

    assert merged.to_attrs() == {"title": "source"}
    assert dropped == {"provider"}
