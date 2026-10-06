from datetime import datetime as dt
from typing import TYPE_CHECKING, assert_type

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geosave_engine.geodata as gs

from geosave_engine.geodata import GeoAnchor, GeoArray, GeoRaster, GeoStack, GeoVector
from geosave_engine.geodata.attrs import CFVariable
from geosave_engine.geodata.utils.geo.geolocator import Place

from tests.geodata.conftest import build_raster


if TYPE_CHECKING:

    def _assert_accessor_types(
        array: gs.DataArray,
        dataset: gs.Dataset,
        tree: gs.DataTree,
    ) -> None:
        assert_type(array.gs, GeoArray)
        assert_type(dataset.gs, GeoRaster)
        assert_type(tree.gs, GeoStack)
        assert_type(dataset.isel(x=0).gs, GeoRaster)


def test_raster_accessor_survives_xarray_selection(raster: xr.Dataset) -> None:
    selected = raster.isel(x=slice(0, 1))

    assert isinstance(selected.gs, GeoRaster)
    assert selected.gs.variables == ("red", "nir")
    geobox = selected.gs.geobox
    assert geobox is not None
    assert geobox.width == 1


def test_stack_accessor_reads_settled_structure(stack: xr.DataTree) -> None:
    assert isinstance(stack.gs, GeoStack)
    assert stack.gs.groups == ("optical", "infrared")


def test_every_xarray_accessor_names_its_variables(
    raster: xr.Dataset, stack: xr.DataTree
) -> None:
    assert raster.red.gs.variables == ("red",)
    assert raster.gs.variables == ("red", "nir")
    assert stack.gs.variables == ("optical/red", "infrared/nir")


def test_unnamed_array_has_no_variable_identity(raster: xr.Dataset) -> None:
    assert raster.red.rename(None).gs.variables == ()


def test_typed_variable_attrs_round_trip(raster: xr.Dataset) -> None:
    metadata = CFVariable(long_name="red reflectance", units="1")

    stamped = raster.gs.rebase(metadata, target="red")

    assert stamped.gs.attrs.data_vars["red"].get(CFVariable) == metadata


def test_unbucketed_time_covers_only_its_own_instants() -> None:
    raster = build_raster(times=2).assign_coords(
        time=np.array(
            ["2025-06-01T10:23:11", "2025-06-11T10:24:02"], dtype="datetime64[ns]"
        )
    )

    start, end = raster.gs.timespan

    assert (start.hour, start.minute, start.second) == (10, 23, 11)
    assert (end.hour, end.minute, end.second) == (10, 24, 2)


def test_timeless_raster_has_no_timespan(raster: xr.Dataset) -> None:
    assert raster.gs.timespan is None


def test_anchor_and_vector_construction() -> None:
    vector = GeoVector.from_geometry("POINT (13 52)")

    anchor = GeoAnchor.from_geometry(vector.gs.footprint, resolution=10)

    assert anchor.geobox.crs is not None
    assert anchor.geobox.crs.to_epsg() == 32633


def test_anchor_formats_place_and_time_through_a_template() -> None:
    anchor = GeoAnchor.from_coordinates(
        52, 13, 500, 10, timespan="2025-06-01/2025-06-11"
    )

    assert (
        anchor.format("{lat:.4f}N_{lon:.4f}E_{start:%Y%m%d}-{end:%Y%m%d}_{res:g}m")
        == "52.0000N_12.9999E_20250601-20250611_10m"
    )
    assert anchor.format("{start:%Y}/{start:%m}/{epsg}") == "2025/06/32633"
    assert anchor.format("{width:.0f}x{height:.0f}") == "5000x5000"


def test_timeless_anchor_refuses_a_date_field() -> None:
    anchor = GeoAnchor.from_coordinates(52, 13, 500, 10)

    with pytest.raises(ValueError, match="no 'start'"):
        anchor.format("{start:%Y}")


def test_anchor_refuses_a_field_it_does_not_offer() -> None:
    anchor = GeoAnchor.from_coordinates(52, 13, 500, 10)

    with pytest.raises(ValueError, match="'place'.*lat.*lon"):
        anchor.format("{place}")


def test_anchor_centroid_reads_longitude_first() -> None:
    longitude, latitude = GeoAnchor.from_coordinates(52, 13, 500, 10).lonlat

    assert longitude == pytest.approx(13, abs=1e-3)
    assert latitude == pytest.approx(52, abs=1e-3)


def test_locate_geocodes_the_centroid(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[float, float]] = []

    def resolve(latitude: float, longitude: float) -> Place:
        asked.append((latitude, longitude))
        return Place(city="Berlin")

    monkeypatch.setattr(Place, "from_coordinate", resolve)

    place = GeoAnchor.from_coordinates(52, 13, 500, 10).locate()

    assert place == Place(city="Berlin")
    assert asked[0] == pytest.approx((52, 13), abs=1e-3)


def test_times_format_as_a_datetime_index_on_a_raster_and_a_band() -> None:
    raster = build_raster(times=2)

    assert isinstance(raster.gs.times, pd.DatetimeIndex)
    assert raster.gs.times.strftime("%Y%m%d").tolist() == ["20250601", "20250602"]
    assert raster.red.gs.times.equals(raster.gs.times)


def test_timeless_data_has_no_times(raster: xr.Dataset) -> None:
    assert raster.gs.times is None
    assert raster.red.gs.times is None


def test_times_leave_chunked_pixels_uncomputed() -> None:
    raster = build_raster(times=2).chunk()

    assert len(raster.gs.times) == 2
    assert isinstance(raster.red.data, da.Array)


def test_odc_geometry_keeps_its_crs() -> None:
    projected = GeoVector.from_geometry(
        "POINT (500000 9000000)",
        crs="EPSG:32749",
    )

    restored = GeoVector.from_geometry(projected.gs.footprint)

    assert restored.crs.to_epsg() == 32749


def test_geodata_types_are_xarray_at_runtime() -> None:
    assert gs.DataArray is xr.DataArray
    assert gs.Dataset is xr.Dataset
    assert gs.DataTree is xr.DataTree


@pytest.mark.parametrize(
    ("labels", "span"),
    [
        (["2018-12-26"], ("2018-12-26T00:00:00", "2018-12-26T23:59:59.999999")),
        (["2018-12-01"], ("2018-12-01T00:00:00", "2018-12-01T23:59:59.999999")),
        (["2018-01-01"], ("2018-01-01T00:00:00", "2018-01-01T23:59:59.999999")),
        (
            ["2018-12-26", "2018-12-27"],
            ("2018-12-26T00:00:00", "2018-12-27T23:59:59.999999"),
        ),
        (
            ["2018-12-26T10:00:00"],
            ("2018-12-26T10:00:00", "2018-12-26T10:00:59.999999"),
        ),
    ],
    ids=["a date", "a month start", "a year start", "two dates", "an hour"],
)
def test_unbucketed_time_covers_what_its_labels_spell(
    labels: list[str], span: tuple[str, str]
) -> None:
    raster = build_raster(times=len(labels)).assign_coords(
        time=np.array(labels, dtype="datetime64[ns]")
    )

    assert raster.gs.timespan == tuple(dt.fromisoformat(edge) for edge in span)


def test_the_accessor_rebases_a_header_and_a_namespace() -> None:
    import xarray as xr

    from geosave_engine.geodata import attrs

    source = xr.Dataset({"red": ("x", [1], {"units": "1"})}, attrs={"title": "S2"})
    target = xr.Dataset({"red": ("x", [2])})

    restored = target.gs.rebase(attrs.create_header(source))
    patched = target.gs.rebase(source.gs.attrs.data_vars["red"], target="red")

    assert restored.attrs == {"title": "S2"}
    assert restored.red.attrs == {"units": "1"}
    assert patched.red.attrs == {"units": "1"}
