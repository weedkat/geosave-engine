from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.raster import raster as build_raster
from geosave_engine.geodata.core.stack import stack as build_stack
import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform.nodata import mask, to_nan

UTM = "EPSG:32633"
CLEAR = [4, 5, 6, 7]


def utm_box() -> GeoBox:
    """Build a small projected grid."""
    return GeoBox.from_bbox((0, 0, 40, 40), crs=UTM, resolution=10)


def scene(*, times: int = 0, nodata: float | int | None = 0) -> xr.Dataset:
    """Build a raster whose pixels all read 7, optionally over a time axis."""
    box = utm_box()
    if times:
        return build_raster(
            {"red": np.full((times, *box.shape), 7, "uint16")},
            box,
            nodata=nodata,
            time=np.array(["2025-06-01", "2025-06-11"][:times], "datetime64[ns]"),
        )
    return build_raster(
        {
            "red": np.full(box.shape, 7, "uint16"),
            "nir": np.full(box.shape, 8, "uint16"),
        },
        box,
        nodata=nodata,
    )


def classification() -> xr.DataArray:
    """Build a scene classification band, half ground and half cloud."""
    dims = scene().gs.grid_dims
    return xr.DataArray(np.array([[4, 4, 9, 9]] * 4), dims=dims)


def test_masking_writes_the_declared_fill_without_promoting_the_dtype() -> None:
    masked = mask(scene(), classification().isin(CLEAR))

    assert masked.red.dtype == np.dtype("uint16")
    assert list(masked.red.values[0]) == [7, 7, 0, 0]


def test_masking_keeps_the_grid_and_the_variable_attrs() -> None:
    source = scene()

    masked = mask(source, classification().isin(CLEAR))

    assert masked.gs.geobox == source.gs.geobox
    assert masked.red.attrs["_FillValue"] == 0


def test_a_flat_mask_covers_every_step_of_a_time_series() -> None:
    masked = mask(scene(times=2), classification().isin(CLEAR))

    assert masked.red.shape == (2, 4, 4)
    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_masking_a_band_returns_a_band() -> None:
    masked = mask(scene().red, classification().isin(CLEAR))

    assert isinstance(masked, xr.DataArray)
    assert list(masked.values[0]) == [7, 7, 0, 0]


def test_masking_a_stack_masks_every_group() -> None:
    source = build_stack({"optical": scene(), "other": scene()})

    masked = mask(source, classification().isin(CLEAR))

    assert masked.gs.groups == ("optical", "other")
    for group in masked.gs.rasters.values():
        assert list(group.red.values[0]) == [7, 7, 0, 0]


def test_a_non_boolean_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        mask(scene(), classification())


def test_a_variable_carrying_no_fill_is_refused_rather_than_promoted() -> None:
    with pytest.raises(ValueError, match="carries no fill value"):
        mask(scene(nodata=None), classification().isin(CLEAR))


def test_a_given_fill_is_written_and_declared() -> None:
    masked = mask(scene(nodata=None), classification().isin(CLEAR), fill=255)

    assert list(masked.red.values[0]) == [7, 7, 255, 255]
    assert masked.red.attrs["_FillValue"] == 255
    assert masked.red.dtype == np.dtype("uint16")


def test_a_given_fill_overrides_what_the_variable_declared() -> None:
    masked = mask(scene(), classification().isin(CLEAR), fill=99)

    assert list(masked.red.values[0]) == [7, 7, 99, 99]
    assert masked.red.attrs["_FillValue"] == 99


@pytest.mark.parametrize("fill", [-1, 70000])
def test_a_fill_the_dtype_cannot_hold_is_refused_rather_than_wrapped(
    fill: int,
) -> None:
    with pytest.raises(ValueError, match="marks no pixel"):
        mask(scene(nodata=None), classification().isin(CLEAR), fill=fill)


def test_a_nan_fill_is_refused_for_an_integer_variable() -> None:
    with pytest.raises(ValueError, match="marks no pixel"):
        mask(scene(nodata=None), classification().isin(CLEAR), fill=float("nan"))


def test_a_nan_fill_is_accepted_by_a_float_variable() -> None:
    box = utm_box()
    floating = build_raster({"elevation": np.ones(box.shape, "float32")}, box)

    masked = mask(floating, classification().isin(CLEAR), fill=float("nan"))

    assert np.isnan(masked.elevation.values[0][-1])


def test_a_bare_numpy_mask_is_read_as_the_grid() -> None:
    bare = np.array([[True, True, False, False]] * 4)

    masked = mask(scene(), bare)

    assert list(masked.red.values[0]) == [7, 7, 0, 0]
    assert masked.red.dtype == np.dtype("uint16")


def test_a_bare_numpy_mask_covers_every_step_of_a_time_series() -> None:
    bare = np.array([[True, True, False, False]] * 4)

    masked = mask(scene(times=2), bare)

    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_a_bare_numpy_mask_off_the_grid_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="names no axes"):
        mask(scene(), np.ones((3, 3), bool))


def test_a_mask_that_misses_the_grid_is_refused() -> None:
    off_grid = xr.DataArray(np.array([True, False]), dims="depth")

    with pytest.raises(ValueError, match="does not cover the grid"):
        mask(scene(), off_grid)


def test_a_mask_spanning_an_axis_the_raster_lacks_is_refused() -> None:
    dims = scene().gs.grid_dims
    extra = xr.DataArray(np.ones((2, 4, 4), bool), dims=("depth", *dims))

    with pytest.raises(ValueError, match="does not carry"):
        mask(scene(), extra)


def test_masking_refuses_something_that_is_not_an_xarray_object() -> None:
    with pytest.raises(TypeError, match="DataArray, Dataset, or DataTree"):
        mask(np.zeros((4, 4)), classification().isin(CLEAR))  # type: ignore[type-var]


def test_cropping_with_a_mask_keeps_the_dtype_and_declared_fill() -> None:
    import geopandas as gpd
    import shapely

    from geosave_engine.geodata.core.vector import GeoVector

    source = scene()
    half = GeoVector(gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 20, 40)], crs=UTM))

    cropped = source.gs.crop(half, mask=True)

    assert cropped.red.dtype == np.dtype("uint16")
    assert cropped.red.attrs["_FillValue"] == 0
    # odc's own apply_mask writes NaN, promoting the variable to float.
    from odc.geo.geom import box as geom_box

    assert source.odc.crop(
        geom_box(0, 0, 20, 40, crs=UTM), apply_mask=True
    ).red.dtype == np.dtype("float32")


def test_blanking_leaves_the_source_alone() -> None:
    # xarray's mask coder pops what it reads, so it must be handed a copy.
    source = scene()
    source["red"][0, 0] = 0

    blanked = to_nan(source)

    assert source.red.attrs["_FillValue"] == 0
    assert source.red.dtype == np.dtype("uint16")
    assert np.isnan(blanked.red.values[0, 0])
    assert np.isnan(to_nan(source).red.values[0, 0])


def test_blanking_leaves_the_packing_alone() -> None:
    packed = scene().gs.rebase(attrs.Packing(scale_factor=1e-4), target="red")

    assert to_nan(packed).red.attrs["scale_factor"] == 1e-4
