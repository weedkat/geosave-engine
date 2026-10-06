from __future__ import annotations

import pytest
import xarray as xr

from geosave_engine.geodata.attrs import (
    ACDD,
    CFVariable,
    MUST_AGREE,
    MODELS,
    AttrsHeader,
    AttrsNamespace,
    DroppedAttr,
    GDALVariable,
    Legend,
    Nodata,
    StackedAttrs,
    TimeSpec,
    create_header,
    merge,
    rebase,
)
from geosave_engine.geodata.warnings import DroppedAttrsWarning


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


def test_only_the_accumulating_model_writes_its_own_merge() -> None:
    stating_a_policy = {
        model.__name__
        for models in MODELS.values()
        for model in models
        if "merge" in model.__dict__
    }

    assert stating_a_policy == {"StacMetadata"}


def test_a_model_keeps_what_the_objects_agree_on_and_drops_the_rest() -> None:
    agreed, dropped = ACDD.merge([ACDD(title="S2"), ACDD(title="S2")])
    assert (agreed.title, dropped) == ("S2", set())

    conflicting, dropped = ACDD.merge([ACDD(title="S2"), ACDD(title="L8")])
    assert conflicting.title is None
    assert dropped == {"title"}


def test_one_field_writes_every_spelling_it_declares() -> None:
    assert Nodata.field_keys["fill_value"] == ("_FillValue", "nodata")
    assert Nodata(fill_value=0).to_attrs() == {"_FillValue": 0, "nodata": 0}


def test_either_spelling_alone_names_the_absent_pixels() -> None:
    cf = AttrsNamespace.from_attrs({"_FillValue": -9999}, "variable")
    odc = AttrsNamespace.from_attrs({"nodata": -9999}, "variable")

    assert cf.get(Nodata).fill_value == -9999
    assert odc.get(Nodata) == cf.get(Nodata)


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
    first = AttrsNamespace.from_attrs({"title": "shared", "note": "a"}, "dataset")
    second = AttrsNamespace.from_attrs({"title": "shared", "note": "b"}, "dataset")

    namespace, dropped = AttrsNamespace.merge([first, second])

    assert namespace.to_attrs() == {"title": "shared"}
    assert dropped == {"note"}


def test_merging_variables_drops_their_per_variable_gdal_identity() -> None:
    red = AttrsNamespace.from_attrs(
        {"variable_name": "B04", "colorinterp": "red"}, "variable"
    )
    green = AttrsNamespace.from_attrs(
        {"variable_name": "B03", "colorinterp": "green"}, "variable"
    )

    namespace, _ = AttrsNamespace.merge([red, green])

    assert namespace.get(GDALVariable) == GDALVariable(
        variable_name=None, colorinterp=None
    )
    assert namespace.to_attrs() == {}


def test_merging_one_variable_keeps_its_gdal_identity() -> None:
    variable = AttrsNamespace.from_attrs(
        {"variable_name": "B04", "colorinterp": "red"}, "variable"
    )

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
        AttrsNamespace(
            models={ACDD.NAME: ACDD(title="x")}, foreign={"title": "untyped"}
        )


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
    unwritable = AttrsNamespace(
        models={StackedAttrs.NAME: StackedAttrs(dataset_attrs={"handle": object()})},
    )
    header = AttrsHeader(
        root=AttrsNamespace.from_attrs({"title": "changed"}, "dataset"),
        coords={"x": unwritable},
    )

    with pytest.raises(TypeError, match="no JSON representation"):
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


def test_rebase_header_leaves_variables_it_does_not_name() -> None:
    header = AttrsHeader.from_attrs(data_vars={"red": {"units": "1"}})
    target = xr.Dataset(
        {"red": ("x", [1], {"stale": True}), "nir": ("x", [2], {"units": "1"})}
    )

    restored = rebase(target, header)

    assert restored.red.attrs == {"units": "1"}
    assert restored.nir.attrs == {"units": "1"}


def test_rebase_refuses_a_model_outside_the_target_scope() -> None:
    ds = xr.Dataset({"red": ("time", [1])}, coords={"time": [0]})

    with pytest.raises(ValueError, match="time_spec belongs on a coordinate"):
        rebase(ds, TimeSpec.from_resample("MS"), target="red")
    with pytest.raises(ValueError, match="nodata belongs on a variable"):
        rebase(ds, nodata={"fill_value": 0})


def test_rebase_refuses_a_namespace_from_another_scope_before_writing() -> None:
    ds = xr.Dataset({"red": ("x", [1], {"units": "1"})})

    with pytest.raises(ValueError, match="belongs on a variable"):
        rebase(ds, create_header(ds).data_vars["red"], inplace=True)
    assert ds.red.attrs == {"units": "1"}


def test_rebase_refuses_a_header_of_another_kind() -> None:
    array = xr.DataArray([1], dims="x", attrs={"nodata": 0})

    with pytest.raises(ValueError, match="belongs on a variable"):
        rebase(xr.Dataset({"red": ("x", [1])}), create_header(array))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"nodata": 0}, {"nodata": -1}),
        ({"nodata": 0}, {}),
        ({"scale_factor": 1e-4}, {"scale_factor": 2e-4}),
        ({"units": "1"}, {"units": "K"}),
        ({"flag_values": [0, 1], "flag_meanings": "a b"}, {}),
    ],
)
def test_merge_refuses_decoding_fields_that_disagree(first, second) -> None:
    rasters = [xr.Dataset({"red": ("x", [1], attrs)}) for attrs in (first, second)]

    with pytest.raises(ValueError, match="must agree") as raised:
        merge(rasters)
    assert "in 'red'" in raised.value.__notes__


def test_merge_drops_a_label_that_disagrees() -> None:
    rasters = [
        xr.Dataset({"red": ("x", [1], {"long_name": name})}) for name in ("a", "b")
    ]

    with pytest.warns(DroppedAttrsWarning, match="red.long_name"):
        header = merge(rasters)
    assert header.data_vars["red"].to_attrs() == {}


def test_merge_refuses_a_dataarray_with_a_dataset() -> None:
    array = xr.DataArray([1], dims="x", name="red", attrs={"long_name": "Red"})

    with pytest.raises(ValueError, match="different scopes") as raised:
        merge([array, xr.Dataset({"red": ("x", [1])}, attrs={"title": "x"})])
    assert "in the objects' own attrs" in raised.value.__notes__


def test_every_decoding_field_carries_the_marker() -> None:
    marked = {
        (model.NAME, name)
        for models in MODELS.values()
        for model in models
        for name, field in model.model_fields.items()
        if MUST_AGREE in field.metadata
    }

    assert marked == {
        ("nodata", "fill_value"),
        ("packing", "scale_factor"),
        ("packing", "add_offset"),
        ("cf_variable", "standard_name"),
        ("cf_variable", "units"),
        ("cf_variable", "cell_methods"),
        ("legend", "flag_values"),
        ("legend", "flag_masks"),
        ("legend", "flag_meanings"),
    }


def test_a_header_without_a_root_leaves_the_root_alone() -> None:
    ds = xr.Dataset({"red": ("x", [1])}, attrs={"title": "S2"})

    restored = rebase(ds, AttrsHeader.from_attrs(data_vars={"red": {"units": "1"}}))

    assert restored.attrs == {"title": "S2"}
    assert restored.red.attrs == {"units": "1"}


def test_merging_root_less_headers_carries_no_root() -> None:
    header = AttrsHeader.from_attrs(data_vars={"red": {"units": "1"}})

    merged, _ = AttrsHeader.merge([header, header])

    assert merged.root.to_attrs() == {}


def test_rebase_header_write_failure_leaves_every_attr_untouched() -> None:
    ds = xr.Dataset({"red": ("x", [1], {"units": "1"})}, attrs={"title": "kept"})
    header = AttrsHeader(
        root=AttrsNamespace(models={ACDD.NAME: ACDD(title="changed")}),
        data_vars={"missing": AttrsNamespace()},
    )

    with pytest.raises(ValueError, match="neither a variable nor a coordinate"):
        rebase(ds, header, inplace=True)
    assert ds.attrs == {"title": "kept"}
    assert ds.red.attrs == {"units": "1"}


def test_rebase_keyword_none_unsets_the_models_keys() -> None:
    ds = xr.Dataset({"red": ("x", [1], {"_FillValue": 0, "nodata": 0, "units": "1"})})

    restored = rebase(ds, target="red", nodata=None)

    assert restored.red.attrs == {"units": "1"}
    assert ds.red.attrs["nodata"] == 0


def test_rebase_keyword_field_none_unsets_only_that_fields_keys() -> None:
    ds = xr.Dataset({"red": ("x", [1], {"variable_name": "B04", "colorinterp": "red"})})

    restored = rebase(ds, target="red", gdal_variable={"colorinterp": None})

    assert restored.red.attrs == {"variable_name": "B04"}
    assert ds.red.attrs["colorinterp"] == "red"


def test_dropped_attr_from_keys_qualifies_each_key() -> None:
    assert {str(attr) for attr in DroppedAttr.from_keys("red", ["a", "b"])} == {
        "red.a",
        "red.b",
    }
    assert {str(attr) for attr in DroppedAttr.from_keys(None, ["a"])} == {"a"}


def test_a_field_set_to_none_writes_no_key() -> None:
    namespace = AttrsNamespace(models={Nodata.NAME: Nodata(fill_value=None)})

    assert Nodata(fill_value=None).to_attrs() == {}
    assert namespace.to_attrs() == {}
    assert namespace.scope == "variable"
    assert AttrsNamespace().scope is None


def test_a_merged_model_leaves_a_disagreed_field_unset_and_names_its_keys() -> None:
    merged, dropped = CFVariable.merge([CFVariable(long_name="Red"), CFVariable()])

    assert merged.long_name is None
    assert merged.to_attrs() == {}
    assert dropped == {"long_name"}


def test_rebase_keyword_none_is_refused_outside_the_models_scope() -> None:
    ds = xr.Dataset({"red": ("time", [1])}, coords={"time": [0]})

    with pytest.raises(ValueError, match="time_spec belongs on a coordinate"):
        rebase(ds, time_spec=None, target="red")
    with pytest.raises(ValueError, match="nodata belongs on a variable"):
        rebase(ds, nodata=None)


def test_rebase_gives_each_target_its_own_values() -> None:
    ds = xr.Dataset({"red": ("x", [1]), "nir": ("x", [1])})
    legend = Legend(flag_values=[0, 1], flag_meanings="a b")

    tagged = rebase(ds, legend, target=["red", "nir"])
    tagged.red.attrs["flag_values"].append(2)

    assert tagged.nir.attrs["flag_values"] == [0, 1]


def test_rebase_inplace_failure_writes_nothing() -> None:
    ds = xr.Dataset({"red": ("x", [1])})

    with pytest.raises(ValueError, match="neither a variable nor a coordinate"):
        rebase(ds, Nodata(fill_value=0), target=["red", "missing"], inplace=True)
    assert ds.red.attrs == {}
