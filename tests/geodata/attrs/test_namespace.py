"""One flat namespace must express one value for each shared metadata key."""

import pytest

from geosave_engine.geodata.attrs import AttrsNamespace, CFCoordinate, CFVariable


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("units", ["kilometre", None])
def test_conflicting_shared_values_do_not_depend_on_model_order(reverse, units):
    models = [CFVariable(units="metre"), CFCoordinate(units=units)]
    if reverse:
        models.reverse()
    namespace = AttrsNamespace(models={model.NAME: model for model in models})

    with pytest.raises(ValueError, match="units"):
        namespace.to_attrs()


def test_editing_a_shared_field_cannot_be_silently_overwritten():
    namespace = AttrsNamespace.from_attrs({"units": "metre"})
    namespace.get(CFVariable).units = "kilometre"

    with pytest.raises(ValueError, match="units"):
        namespace.to_attrs()


@pytest.mark.parametrize("units", ["metre", None])
def test_agreeing_shared_values_keep_foreign_null_values(units):
    namespace = AttrsNamespace(
        models={"cf": CFVariable(units=units), "coordinate": CFCoordinate(units=units)},
        foreign={"provider_note": None},
    )

    expected = {"provider_note": None}
    if units is not None:
        expected["units"] = units
    assert namespace.to_attrs() == expected


def test_unset_shared_field_is_not_an_explicit_clear():
    namespace = AttrsNamespace(
        models={"cf": CFVariable(units="metre"), "coordinate": CFCoordinate(axis="X")}
    )

    assert namespace.to_attrs() == {"units": "metre", "axis": "X"}


def test_equal_flat_namespaces_merge_with_different_shared_model_presence():
    from geosave_engine.geodata.attrs import Nodata
    from geosave_engine.geodata.attrs import model as registry

    class OtherNodata(Nodata):
        NAME = "test_shared_nodata"

    try:
        first = AttrsNamespace(models={"nodata": Nodata(fill_value=0)})
        second = AttrsNamespace(
            models={
                "nodata": Nodata(fill_value=0),
                OtherNodata.NAME: OtherNodata(fill_value=0),
            }
        )
        merged, dropped = AttrsNamespace.merge([first, second])

        assert merged.to_attrs() == {"_FillValue": 0, "nodata": 0}
        assert dropped == set()
    finally:
        registry._MODEL_TYPES.pop(OtherNodata.NAME)
