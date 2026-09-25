from __future__ import annotations

from typing import ClassVar

import pytest
import xarray as xr

from geosave_engine.geodata.attrs import (
    ACDD,
    REGISTERED_MODELS,
    AttrsHeader,
    AttrsModel,
    AttrsNamespace,
    CFCoordinate,
    CFVariable,
    GDALVariable,
    Legend,
    Nodata,
    create_header,
    merge,
    rebase,
)
from geosave_engine.geodata.errors import DroppedAttrsWarning


def _source() -> xr.Dataset:
    return xr.Dataset(
        {"red": ("x", [1, 2], {"_FillValue": 0, "source_band": "B04"})},
        coords={"x": ("x", [0, 1], {"axis_note": "east"})},
        attrs={"title": "source", "provider_id": "scene-001"},
    )


def test_namespace_get_accepts_a_model_class_or_registered_name() -> None:
    model = ACDD(title="source")
    namespace = AttrsNamespace(models={ACDD.NAME: model})

    assert namespace.get(ACDD) is model
    assert namespace.get(ACDD.NAME) is model


def test_namespace_owns_flat_attrs_lifecycle() -> None:
    first = AttrsNamespace.from_attrs({"title": "source", "provider": "one"})
    second = AttrsNamespace.from_attrs({"title": "source", "provider": "two"})

    merged, dropped = AttrsNamespace.merge([first, second])

    assert merged.to_attrs() == {"title": "source"}
    assert dropped == {"provider"}


def test_only_the_accumulating_model_writes_its_own_merge() -> None:
    stating_a_policy = {
        model.__name__
        for model in REGISTERED_MODELS.values()
        if "merge" in model.__dict__
    }

    assert stating_a_policy == {"StacMetadata"}


def test_a_model_keeps_what_the_objects_agree_on_and_drops_the_rest() -> None:
    agreed, dropped = Nodata.merge([Nodata(fill_value=0), Nodata(fill_value=0)])
    assert (agreed.fill_value, dropped) == (0, set())

    conflicting, dropped = Nodata.merge([Nodata(fill_value=0), Nodata(fill_value=-1)])
    assert conflicting.fill_value is None
    assert dropped == {"_FillValue", "nodata"}


def test_one_field_writes_every_spelling_it_declares() -> None:
    assert Nodata.field_keys["fill_value"] == ("_FillValue", "nodata")
    assert Nodata(fill_value=0).to_attrs() == {"_FillValue": 0, "nodata": 0}


@pytest.mark.parametrize(
    ("suffix", "keys"),
    [
        ("string", "stored"),
        ("non_string", ("stored", 1)),
        ("empty_string", ("",)),
        ("empty", ()),
    ],
)
def test_field_keys_are_nonempty_tuples_of_attr_names(suffix, keys) -> None:
    from geosave_engine.geodata.attrs import model as registry

    model_name = f"test_bad_field_keys_{suffix}"
    try:
        with pytest.raises(ValueError, match="field_keys"):

            class BadFieldKeys(AttrsModel):
                NAME: ClassVar[str] = model_name
                field_keys: ClassVar = {"value": keys}

                value: int = 0
    finally:
        registry._MODEL_TYPES.pop(model_name, None)
        for attr_key, owner in list(registry._FIELD_BY_ATTR_KEY.items()):
            if owner[0].NAME == model_name:
                del registry._FIELD_BY_ATTR_KEY[attr_key]


def test_either_spelling_alone_names_the_absent_pixels() -> None:
    cf = AttrsNamespace.from_attrs({"_FillValue": -9999})
    odc = AttrsNamespace.from_attrs({"nodata": -9999})

    assert cf.get(Nodata).fill_value == -9999
    assert odc.get(Nodata) == cf.get(Nodata)


def test_spellings_set_to_different_values_are_refused() -> None:
    with pytest.raises(ValueError, match="they spell one Nodata.fill_value"):
        AttrsNamespace.from_attrs({"_FillValue": 0, "nodata": -9999})


def test_rebase_header_round_trips_typed_and_foreign_attrs() -> None:
    header = create_header(_source())
    target = xr.Dataset(
        {"red": ("x", [3, 4], {"_FillValue": -1, "stale": True})},
        coords={"x": ("x", [0, 1], {"stale": True})},
        attrs={"title": "stale", "stale": True},
    )

    stamped = rebase(target, header)

    assert stamped.attrs == {"title": "source", "provider_id": "scene-001"}
    assert stamped.red.attrs == {"_FillValue": 0, "nodata": 0, "source_band": "B04"}
    assert stamped.x.attrs == {"axis_note": "east"}


def test_merge_drops_disagreeing_foreign_attrs_before_rebase() -> None:
    first = _source()
    second = _source()
    second.attrs["provider_id"] = "scene-002"
    second.red.attrs["source_band"] = "B08"
    second.x.attrs["axis_note"] = "west"

    with pytest.warns(DroppedAttrsWarning, match="carried differently"):
        header = merge([first, second])
    stamped = rebase(first, header)

    assert "provider_id" not in stamped.attrs
    assert "source_band" not in stamped.red.attrs
    assert "axis_note" not in stamped.x.attrs


def test_merge_keeps_agreeing_foreign_attrs() -> None:
    header = merge([_source(), _source()])

    assert header.root.foreign == {"provider_id": "scene-001"}
    assert header.data_vars["red"].foreign == {"source_band": "B04"}
    assert header.coords["x"].foreign == {"axis_note": "east"}


def test_merging_namespaces_keeps_only_what_they_carry_alike() -> None:
    first = AttrsNamespace.from_attrs({"title": "shared", "note": "a"})
    second = AttrsNamespace.from_attrs({"title": "shared", "note": "b"})

    namespace, dropped = AttrsNamespace.merge([first, second])

    assert namespace.to_attrs() == {"title": "shared"}
    assert dropped == {"note"}


def test_merging_variables_drops_their_per_variable_gdal_identity() -> None:
    red = AttrsNamespace.from_attrs({"variable_name": "B04", "colorinterp": "red"})
    green = AttrsNamespace.from_attrs({"variable_name": "B03", "colorinterp": "green"})

    namespace, _ = AttrsNamespace.merge([red, green])

    assert namespace.get(GDALVariable) == GDALVariable(
        variable_name=None, colorinterp=None
    )
    assert namespace.to_attrs() == {}


def test_merging_one_variable_keeps_its_gdal_identity() -> None:
    variable = AttrsNamespace.from_attrs({"variable_name": "B04", "colorinterp": "red"})

    namespace, dropped = AttrsNamespace.merge([variable])

    assert namespace.get(GDALVariable) == GDALVariable(
        variable_name="B04", colorinterp="red"
    )
    assert dropped == set()


def test_header_merge_drops_conflicting_gdal_identity() -> None:
    first = rebase(
        _source(), GDALVariable(variable_name="B04", colorinterp="red"), target="red"
    )
    second = rebase(
        _source(), GDALVariable(variable_name="B08", colorinterp="nir"), target="red"
    )

    with pytest.warns(DroppedAttrsWarning, match="red.variable_name"):
        header = merge([first, second])

    assert header.data_vars["red"].get(GDALVariable).variable_name is None


def test_rebase_header_prevalidates_targets_before_an_inplace_write() -> None:
    header = create_header(_source())
    target = xr.Dataset({"red": ("y", [3, 4])}, attrs={"stale": True})

    with pytest.raises(ValueError, match="'x' is neither"):
        rebase(target, header, inplace=True)

    assert target.attrs == {"stale": True}


def test_a_namespace_rejects_foreign_keys_owned_by_a_model() -> None:
    with pytest.raises(ValueError, match="collide"):
        AttrsNamespace(foreign={"title": "untyped"})


def test_a_namespace_rejects_a_model_filed_under_another_name() -> None:
    with pytest.raises(TypeError, match="must be ACDD"):
        AttrsNamespace(models={ACDD.NAME: Nodata(fill_value=0)})


def test_rebase_rejects_target_alongside_a_header() -> None:
    header = AttrsHeader(root=AttrsNamespace(models={ACDD.NAME: ACDD(title="source")}))

    with pytest.raises(ValueError, match="target"):
        rebase(xr.Dataset({"red": ("x", [1, 2])}), header, target="red")  # type: ignore[call-overload]


def test_rebase_rejects_keyword_models_alongside_a_header() -> None:
    header = AttrsHeader(root=AttrsNamespace(models={ACDD.NAME: ACDD(title="source")}))

    with pytest.raises(ValueError, match="keyword models"):
        rebase(xr.Dataset(), header, acdd={"title": "override"})  # type: ignore[call-overload]


def test_rebase_namespace_patches_target_leaving_other_keys_alone() -> None:
    ds = xr.Dataset(
        {"red": ("x", [1, 2], {"_FillValue": -1, "stale": True, "kept": True})}
    )
    namespace = AttrsNamespace(
        models={Nodata.NAME: Nodata(fill_value=0)}, foreign={"note": "typed"}
    )

    patched = rebase(ds, namespace, target="red")  # type: ignore[call-overload]

    assert patched.red.attrs == {
        "_FillValue": 0,
        "nodata": 0,
        "stale": True,
        "kept": True,
        "note": "typed",
    }


def test_rebase_namespace_patches_root_when_target_is_none() -> None:
    ds = xr.Dataset(attrs={"stale": True, "title": "old"})
    namespace = AttrsNamespace(models={ACDD.NAME: ACDD(title="new")})

    patched = rebase(ds, namespace)  # type: ignore[call-overload]

    assert patched.attrs == {"stale": True, "title": "new"}


def test_rebase_rejects_keyword_models_alongside_a_namespace() -> None:
    namespace = AttrsNamespace(models={ACDD.NAME: ACDD(title="source")})

    with pytest.raises(ValueError, match="keyword models"):
        rebase(xr.Dataset(), namespace, acdd={"title": "override"})  # type: ignore[call-overload]


def test_rebase_header_preserves_registered_model_serialization() -> None:
    header = AttrsHeader(
        data_vars={"red": AttrsNamespace(models={Nodata.NAME: Nodata(fill_value=0)})}
    )
    stamped = rebase(xr.Dataset({"red": ("x", [1, 2])}), header)

    # Nodata mirrors the fill across odc's spelling too, so both keys land.
    assert stamped.red.attrs == {"_FillValue": 0, "nodata": 0}


@pytest.mark.parametrize("as_namespace", [False, True])
def test_invalid_later_target_leaves_all_attrs_unchanged(as_namespace: bool) -> None:
    ds = _source()
    before = ds.copy(deep=True)
    model = Nodata(fill_value=255)
    patch = AttrsNamespace(models={model.NAME: model}) if as_namespace else model

    with pytest.raises(ValueError, match="missing"):
        rebase(ds, patch, target=["red", "missing"], inplace=True)

    xr.testing.assert_identical(ds, before)


def test_header_serialization_failure_leaves_all_attrs_unchanged() -> None:
    ds = _source()
    before = ds.copy(deep=True)
    conflicting = AttrsNamespace(
        models={"cf": CFVariable(units="m"), "coordinate": CFCoordinate(units="km")}
    )
    header = AttrsHeader(
        root=AttrsNamespace.from_attrs({"title": "changed"}),
        data_vars={"red": conflicting},
    )

    with pytest.raises(ValueError, match="units"):
        rebase(ds, header, inplace=True)

    xr.testing.assert_identical(ds, before)


def test_ordered_model_edits_remain_explicit_overrides() -> None:
    ds = _source()

    result = rebase(
        ds,
        Nodata(fill_value=255),
        Nodata(fill_value=None),
        target="red",
        nodata={"fill_value": 10},
    )

    assert result.red.attrs == {"_FillValue": 10, "nodata": 10, "source_band": "B04"}
    assert ds.red.attrs == {"_FillValue": 0, "source_band": "B04"}


def test_keyword_none_clears_every_key_owned_by_the_model() -> None:
    source = xr.Dataset(
        {"red": ("x", [1], {"_FillValue": 0, "nodata": 0, "kept": True})}
    )

    result = rebase(source, target="red", nodata=None)

    assert result.red.attrs == {"kept": True}


def test_metadata_edits_keep_lazy_pixels_and_coordinate_metadata() -> None:
    import dask.array as da
    from dask.callbacks import Callback

    ds = _source().chunk({"x": 1})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = rebase(ds, Nodata(fill_value=255), target="red")

    assert tasks == []
    assert isinstance(result.red.data, da.Array)
    assert result.red.data is ds.red.data
    xr.testing.assert_identical(result.x, ds.x)


def test_bulk_model_targets_keep_independent_mutable_metadata() -> None:
    ds = xr.Dataset({"red": ("x", [0, 1]), "nir": ("x", [0, 1])})
    legend = Legend(flag_values=[0, 1], flag_meanings="water land")
    result = rebase(ds, legend, target=["red", "nir"])

    result.red.attrs["flag_values"].append(2)

    assert result.nir.attrs["flag_values"] == [0, 1]
    assert legend.flag_values == [0, 1]
