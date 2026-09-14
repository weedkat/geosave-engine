from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.raster import raster as build_raster
from geosave_engine.geodata.transform.nodata import is_fill, to_nan
from geosave_engine.geodata.transform.packing import unpack

UTM = "EPSG:32633"


def utm_box() -> GeoBox:
    """Build a small projected grid."""
    return GeoBox.from_bbox((300000, 5000000, 300020, 5000020), crs=UTM, resolution=10)


def stored(
    value: int = 1000,
    *,
    scale: float | None = 1e-4,
    offset: float | None = None,
    nodata: int | None = None,
    dtype: str = "uint16",
) -> xr.Dataset:
    """Build a raster whose `red` holds `value` as a stored digital number."""
    box = utm_box()
    raster = build_raster({"red": np.full(box.shape, value, dtype)}, box, nodata=nodata)
    packing = {}
    if scale is not None:
        packing["scale_factor"] = scale
    if offset is not None:
        packing["add_offset"] = offset
    if not packing:
        return raster
    return raster.gs.rebase(attrs.Packing(**packing), target="red")


def test_stored_numbers_read_out_as_physical_values() -> None:
    physical = unpack(stored(1000, scale=1e-4))

    assert physical.red.values[0, 0] == pytest.approx(0.1)


def test_an_offset_is_added_after_the_scale() -> None:
    physical = unpack(stored(1000, scale=1e-4, offset=5.0))

    assert physical.red.values[0, 0] == pytest.approx(5.1)


def test_an_offset_alone_shifts_without_scaling() -> None:
    physical = unpack(stored(1000, scale=None, offset=5.0))

    assert physical.red.values[0, 0] == pytest.approx(1005.0)


def test_a_variable_carrying_no_packing_passes_through_untouched() -> None:
    plain = stored(1000, scale=None)

    physical = unpack(plain)

    assert physical.red.values[0, 0] == 1000
    assert physical.red.dtype == np.dtype("uint16")


def test_the_packing_leaves_attrs_with_the_values_it_described() -> None:
    physical = unpack(stored(1000, scale=1e-4, offset=5.0))

    assert "scale_factor" not in physical.red.attrs
    assert "add_offset" not in physical.red.attrs


def test_unpacking_leaves_the_source_alone() -> None:
    # xarray's scale coder pops what it reads, so it must be handed a copy.
    source = stored(1000, scale=1e-4)

    first = unpack(source)

    assert source.red.attrs["scale_factor"] == pytest.approx(1e-4)
    assert source.red.dtype == np.dtype("uint16")
    assert first.red.values[0, 0] == pytest.approx(unpack(source).red.values[0, 0])


def test_other_attrs_and_coordinates_ride_through() -> None:
    source = stored(1000, scale=1e-4)
    source.attrs["title"] = "scene"
    source["red"].attrs.update({"long_name": "Red", "units": "1"})

    physical = unpack(source)

    assert physical.attrs["title"] == "scene"
    assert physical.red.attrs["long_name"] == "Red"
    assert physical.gs.geobox == source.gs.geobox


def test_a_variable_that_is_not_packed_rides_alongside_one_that_is() -> None:
    source = stored(1000, scale=1e-4)
    source["dem"] = (source.gs.grid_dims, np.full(utm_box().shape, 5.0, "float32"))

    physical = unpack(source)

    assert sorted(str(name) for name in physical.data_vars) == ["dem", "red"]
    assert physical.dem.values[0, 0] == pytest.approx(5.0)
    assert physical.red.values[0, 0] == pytest.approx(0.1)


def test_a_band_unpacks_as_a_band() -> None:
    physical = unpack(stored(1000, scale=1e-4).red)

    assert isinstance(physical, xr.DataArray)
    assert physical.values[0, 0] == pytest.approx(0.1)


def test_unpacking_stays_lazy_while_its_raster_is() -> None:
    physical = unpack(stored(1000, scale=1e-4).chunk())

    assert physical.red.chunks is not None
    assert physical.red.compute().values[0, 0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("fill", "scale", "offset", "dtype"),
    [
        (0, 1e-4, 5.0, "int32"),
        (65535, 1e-4, None, "uint16"),
        (-9999, 0.01, None, "int32"),
        (0, 1e-4, None, "int32"),
        (0, None, 5.0, "int32"),
    ],
    ids=["offset", "large fill", "negative fill", "fixed point", "offset only"],
)
def test_unpacking_refuses_a_variable_still_marking_its_nodata(
    fill: int, scale: float | None, offset: float | None, dtype: str
) -> None:
    packed = stored(1000, scale=scale, offset=offset, nodata=fill, dtype=dtype)

    with pytest.raises(ValueError, match="marks its nodata pixels"):
        unpack(packed)


def test_blanking_the_nodata_first_keeps_it_apart_from_a_reading() -> None:
    packed = stored(1000, scale=1e-4, offset=5.0, nodata=0, dtype="int32")
    packed["red"][0, 0] = 0

    physical = unpack(to_nan(packed))

    assert np.isnan(physical.red.values[0, 0])
    assert physical.red.values[0, 1] == pytest.approx(5.1)


def test_a_variable_marking_nodata_but_carrying_no_packing_is_left_alone() -> None:
    plain = stored(1000, scale=None, nodata=0, dtype="int32")

    physical = unpack(plain)

    assert physical.red.values[0, 0] == 1000
    assert is_fill(physical.red).values[0, 0] == np.False_
