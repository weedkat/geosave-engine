from __future__ import annotations

import pytest
import xarray as xr

from geosave_engine.geodata.attrs import (
    ACDD,
    AttrsHeader,
    AttrsNamespace,
    CFCoordinate,
    CFVariable,
    GeoTIFFTags,
    Legend,
    Packing,
    RenderHints,
    StacMetadata,
    Tiling,
    TimeSpec,
    combine,
    read,
    rebase,
    stamp,
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

    combined, dropped = AttrsNamespace.combine([first, second])

    assert combined.to_attrs() == {"title": "source"}
    assert dropped == {"provider"}


def test_every_registered_model_declares_its_combine_method() -> None:
    models = (
        ACDD,
        CFCoordinate,
        CFVariable,
        GeoTIFFTags,
        Legend,
        Packing,
        RenderHints,
        StacMetadata,
        Tiling,
        TimeSpec,
    )

    assert all("combine" in model.__dict__ for model in models)


def test_stamp_round_trips_typed_and_foreign_attrs() -> None:
    header = read(_source())
    target = xr.Dataset(
        {"red": ("x", [3, 4], {"_FillValue": -1, "stale": True})},
        coords={"x": ("x", [0, 1], {"stale": True})},
        attrs={"title": "stale", "stale": True},
    )

    stamped = stamp(target, header)

    assert stamped.attrs == {"title": "source", "provider_id": "scene-001"}
    assert stamped.red.attrs == {"_FillValue": 0, "nodata": 0, "source_band": "B04"}
    assert stamped.x.attrs == {"axis_note": "east"}


def test_combine_drops_disagreeing_foreign_attrs_before_stamping() -> None:
    first = _source()
    second = _source()
    second.attrs["provider_id"] = "scene-002"
    second.red.attrs["source_band"] = "B08"
    second.x.attrs["axis_note"] = "west"

    with pytest.warns(DroppedAttrsWarning, match="did not state alike"):
        header = combine([first, second])
    stamped = stamp(first, header)

    assert "provider_id" not in stamped.attrs
    assert "source_band" not in stamped.red.attrs
    assert "axis_note" not in stamped.x.attrs


def test_combine_keeps_agreeing_foreign_attrs() -> None:
    header = combine([_source(), _source()])

    assert header.root.foreign == {"provider_id": "scene-001"}
    assert header.data_vars["red"].foreign == {"source_band": "B04"}
    assert header.coords["x"].foreign == {"axis_note": "east"}


def test_stamp_prevalidates_targets_before_an_inplace_write() -> None:
    header = read(_source())
    target = xr.Dataset({"red": ("y", [3, 4])}, attrs={"stale": True})

    with pytest.raises(ValueError, match="'x' is neither"):
        stamp(target, header, inplace=True)

    assert target.attrs == {"stale": True}


def test_stamp_rejects_foreign_keys_owned_by_a_model() -> None:
    header = AttrsHeader(root=AttrsNamespace(foreign={"title": "untyped"}))

    with pytest.raises(ValueError, match="collide"):
        stamp(xr.Dataset(), header)


def test_rebase_rejects_headers() -> None:
    with pytest.raises(TypeError, match="AttrsModel"):
        rebase(
            xr.Dataset(),
            AttrsHeader(root=AttrsNamespace(models={ACDD.NAME: ACDD(title="source")})),
        )  # type: ignore[arg-type]


def test_stamp_preserves_registered_model_serialization() -> None:
    header = AttrsHeader(
        variables={"red": AttrsNamespace(models={Packing.NAME: Packing(fill_value=0)})},
        var_names=frozenset({"red"}),
    )
    stamped = stamp(xr.Dataset({"red": ("x", [1, 2])}), header)

    # Packing mirrors the fill across odc's spelling too, so both keys land.
    assert stamped.red.attrs == {"_FillValue": 0, "nodata": 0}
