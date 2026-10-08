import json

import numpy as np

from geosave_engine.geodata.attrs import (
    AttrsHeader,
    AttrsNamespace,
    Nodata,
    StackedAttrs,
)


def test_stacking_keeps_dataset_and_variable_namespaces_separate():
    source = AttrsHeader.from_attrs(
        root={"title": "scene", "nodata": -9999},
        data_vars={
            "red": {"nodata": 0, "units": "1", "long_name": "red"},
            "nir": {"_FillValue": 0, "units": "1", "long_name": "nir"},
        },
    )

    shared, preserved = StackedAttrs.from_header(source)

    assert shared.get(Nodata).fill_value == 0
    assert shared.to_attrs() == {"nodata": 0, "_FillValue": 0, "units": "1"}
    assert preserved.dataset_attrs == {"title": "scene", "nodata": -9999}
    assert preserved.variable_attrs == {
        "red": {"long_name": "red"},
        "nir": {"long_name": "nir"},
    }
    restored = preserved.to_header(variables=source.data_vars, shared=shared)
    assert restored == source


def test_differing_variable_fields_are_preserved_without_raising():
    source = AttrsHeader.from_attrs(
        data_vars={
            "red": {"nodata": 0, "scale_factor": 0.1, "units": "1"},
            "nir": {"nodata": -1, "scale_factor": 0.2},
        },
    )

    shared, preserved = StackedAttrs.from_header(source)

    assert shared.to_attrs() == {}
    assert preserved.to_header(variables=source.data_vars, shared=shared) == source


def test_restoration_selects_variables_and_applies_shared_edits():
    source = AttrsHeader.from_attrs(
        root={"title": "scene"},
        data_vars={
            "red": {"nodata": 0, "long_name": "red", "units": "1"},
            "nir": {"nodata": 0, "long_name": "nir", "units": "1"},
        },
    )
    _, preserved = StackedAttrs.from_header(source)
    edited = AttrsNamespace.from_attrs({"nodata": -1, "title": "edited"}, "variable")

    restored = preserved.to_header(variables=["nir"], shared=edited)

    assert restored.root.to_attrs() == {"title": "scene"}
    assert list(restored.data_vars) == ["nir"]
    assert restored.data_vars["nir"].to_attrs() == {
        "long_name": "nir",
        "nodata": -1,
        "_FillValue": -1,
        "title": "edited",
    }


def test_preserved_metadata_restores_from_coordinate_json_text():
    source = AttrsHeader.from_attrs(
        root={"foreign": {"site": 7}},
        data_vars={"red": {"nodata": -1}, "nir": {"nodata": 0}},
    )
    shared, preserved = StackedAttrs.from_header(source)
    stored = {key: json.dumps(value) for key, value in preserved.to_attrs().items()}
    reopened = StackedAttrs.from_attrs(stored)

    restored = reopened.to_header(variables=["red"], shared=shared)

    assert restored.root.to_attrs() == {"foreign": {"site": 7}}
    assert restored.data_vars["red"].get(Nodata).fill_value == -1


def test_differing_nan_nodata_is_preserved_in_native_coordinate_attrs():
    source = AttrsHeader.from_attrs(
        data_vars={"red": {"nodata": np.nan}, "nir": {"nodata": 0}},
    )
    shared, preserved = StackedAttrs.from_header(source)
    reopened = StackedAttrs.from_attrs(preserved.to_attrs())

    restored = reopened.to_header(variables=["red"], shared=shared)

    assert np.isnan(restored.data_vars["red"].get(Nodata).fill_value)
