from __future__ import annotations

import pytest
import xarray as xr

from geosave_engine.geodata.attrs import (
    ACDD,
    AttrsHeader,
    AttrsNamespace,
    CFCoordinate,
    CFVariable,
    GDALVariable,
    GeoTIFFTags,
    Legend,
    Nodata,
    Packing,
    StacMetadata,
    TimeSpec,
    merge,
    read,
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


def test_every_registered_model_declares_its_merge_method() -> None:
    models = (
        ACDD,
        CFCoordinate,
        CFVariable,
        GeoTIFFTags,
        Legend,
        GDALVariable,
        Nodata,
        Packing,
        StacMetadata,
        TimeSpec,
    )

    assert all("merge" in model.__dict__ for model in models)


def test_rebase_header_round_trips_typed_and_foreign_attrs() -> None:
    header = read(_source())
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


def test_merge_flattens_variables_into_one_namespace() -> None:
    first = xr.Variable(("x",), [1, 2], attrs={"title": "shared", "note": "a"})
    second = xr.Variable(("x",), [3, 4], attrs={"title": "shared", "note": "b"})

    with pytest.warns(DroppedAttrsWarning, match="carried differently"):
        namespace = merge([first, second])

    assert namespace.to_attrs() == {"title": "shared"}


def test_merge_variables_uses_action_in_the_warning() -> None:
    first = xr.Variable(("x",), [1, 2], attrs={"note": "a"})
    second = xr.Variable(("x",), [3, 4], attrs={"note": "b"})

    with pytest.warns(DroppedAttrsWarning, match="stacking"):
        merge([first, second], action="stacking")


def test_rebase_header_prevalidates_targets_before_an_inplace_write() -> None:
    header = read(_source())
    target = xr.Dataset({"red": ("y", [3, 4])}, attrs={"stale": True})

    with pytest.raises(ValueError, match="'x' is neither"):
        rebase(target, header, inplace=True)

    assert target.attrs == {"stale": True}


def test_rebase_header_rejects_foreign_keys_owned_by_a_model() -> None:
    header = AttrsHeader(root=AttrsNamespace(foreign={"title": "untyped"}))

    with pytest.raises(ValueError, match="collide"):
        rebase(xr.Dataset(), header)


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
        variables={"red": AttrsNamespace(models={Nodata.NAME: Nodata(fill_value=0)})},
        var_names=frozenset({"red"}),
    )
    stamped = rebase(xr.Dataset({"red": ("x", [1, 2])}), header)

    # Nodata mirrors the fill across odc's spelling too, so both keys land.
    assert stamped.red.attrs == {"_FillValue": 0, "nodata": 0}
