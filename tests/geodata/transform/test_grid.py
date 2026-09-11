from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.transform import grid as transform_grid
from geosave_engine.geodata.utils.io import zarr

from .conftest import UTM, build, geobox


def test_align_cuts_operands_to_their_shared_extent() -> None:
    left = geobox()
    right = geobox((300_160.0, 5_000_160.0, 300_480.0, 5_000_480.0))

    cut_left, cut_right = transform_grid.align([build(left), build(right)])

    assert cut_left.red.shape == (2, 16, 16)
    assert cut_right.red.shape == (2, 16, 16)
    assert cut_left.gs.geobox == cut_right.gs.geobox


def test_align_slices_without_touching_pixels_or_attrs() -> None:
    left = geobox()
    right = geobox((300_160.0, 5_000_160.0, 300_480.0, 5_000_480.0))

    cut, _ = transform_grid.align([build(left), build(right)])

    assert cut.red.dtype == np.dtype("uint16")
    assert int(cut.red.values.max()) == 1000
    assert cut.red.attrs["scale_factor"] == pytest.approx(1e-4)


def test_align_passes_one_raster_through() -> None:
    box = geobox()

    (only,) = transform_grid.align([build(box)])

    assert only.gs.geobox == box


def test_align_refuses_no_rasters() -> None:
    with pytest.raises(ValueError, match="at least one raster"):
        transform_grid.align([])


def test_align_refuses_disjoint_extents() -> None:
    box = geobox()
    far = geobox((400_000.0, 5_100_000.0, 400_320.0, 5_100_320.0))

    with pytest.raises(ValueError, match="share no grid cell"):
        transform_grid.align([build(box), build(far)])


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(geobox(resolution=15.0), id="resolution"),
        pytest.param(
            GeoBox(
                (32, 32),
                Affine(10.0, 0.0, 300_005.0, 0.0, -10.0, 5_000_325.0),
                UTM,
            ),
            id="pixel-phase",
        ),
    ],
)
def test_align_refuses_grids_it_would_have_to_resample(other: GeoBox) -> None:
    with pytest.raises(ValueError, match="not on one pixel grid"):
        transform_grid.align([build(geobox()), build(other)])


def test_align_refuses_mismatched_crs() -> None:
    with pytest.raises(ValueError, match=r"\['EPSG:32633', 'EPSG:32634'\]"):
        transform_grid.align([build(geobox()), build(geobox(crs="EPSG:32634"))])


def test_reproject_lands_on_the_target_crs() -> None:
    warped = transform_grid.reproject(build(geobox()), "EPSG:4326")

    assert warped.gs.crs.epsg == 4326
    assert warped.gs.grid_dims == ("latitude", "longitude")
    assert warped.latitude.attrs["standard_name"] == "latitude"
    assert warped.red.encoding["grid_mapping"] == "spatial_ref"


def test_reproject_preserves_dtype_and_packing() -> None:
    box = geobox()

    coarser = transform_grid.reproject(
        build(box), box.zoom_out(2), resampling="average"
    )

    assert coarser.red.dtype == np.dtype("uint16")
    assert coarser.red.attrs["scale_factor"] == pytest.approx(1e-4)
    assert coarser.red.shape == (2, 16, 16)


def test_reproject_refuses_blending_a_categorical_variable() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="carry a class map"):
        transform_grid.reproject(
            build(box, labelled=True), box.zoom_out(2), resampling="bilinear"
        )


def test_reproject_carries_a_class_map_through_nearest() -> None:
    box = geobox()

    coarser = transform_grid.reproject(build(box, labelled=True), box.zoom_out(2))

    assert coarser.cls.attrs["class_map"] == {0: "bg", 1: "crop"}


def test_reproject_refuses_a_resolution_alongside_a_geobox() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="only when reprojecting onto a CRS"):
        transform_grid.reproject(build(box), box.zoom_out(2), resolution=20.0)


def test_reproject_refuses_a_raster_carrying_no_grid() -> None:
    import xarray as xr

    placeless = xr.Dataset({"red": (("y", "x"), np.ones((4, 4), "uint16"))})

    with pytest.raises(ValueError, match="no locatable grid"):
        transform_grid.reproject(placeless, "EPSG:4326")


def test_reproject_output_can_be_written_to_zarr(tmp_path) -> None:
    # odc names the grid mapping in encoding; the Zarr writer refuses a key
    # held in both encoding and attrs.
    warped = transform_grid.reproject(build(geobox()), "EPSG:4326")

    assert warped.red.encoding["grid_mapping"] == "spatial_ref"
    assert warped.red.encoding["grid_mapping"] == "spatial_ref"
    assert warped.gs.to_zarr(tmp_path / "warped.zarr").exists()


def covered(left: float, value: int, *, fill: int | None = 0) -> xr.Dataset:
    """Build one mosaic operand of distinct pixels on the default grid size.

    Args:
        left: Western edge of the operand's grid, in CRS units.
        value: Value every pixel of the operand carries.
        fill: Fill value the operand declares, or None to declare none.

    Returns:
        Timeless placed raster holding `red` as uint16.
    """
    box = geobox((left, 5_000_000.0, left + 320.0, 5_000_320.0))
    built = raster({"red": np.full(tuple(box.shape), value, "uint16")}, box)
    return built.gs.write_nodata(fill)


def covered_float(
    left: float, value: float, *, fill: float | None = np.nan
) -> xr.Dataset:
    """Build one mosaic operand of distinct float pixels on the default grid size.

    Args:
        left: Western edge of the operand's grid, in CRS units.
        value: Value every pixel of the operand carries.
        fill: Fill value the operand declares, or None to declare none.

    Returns:
        Timeless placed raster holding `red` as float32.
    """
    box = geobox((left, 5_000_000.0, left + 320.0, 5_000_320.0))
    return raster(
        {"red": np.full(tuple(box.shape), value, "float32")}, box
    ).gs.write_nodata(fill)


def test_mosaic_lays_adjacent_rasters_onto_their_union() -> None:
    laid = transform_grid.mosaic([covered(300_000.0, 1), covered(300_320.0, 2)])

    assert laid.gs.geobox.shape == (32, 64)
    assert int(laid.red.values[0, 0]) == 1
    assert int(laid.red.values[0, -1]) == 2


def test_mosaic_holds_the_fill_where_no_raster_reaches() -> None:
    laid = transform_grid.mosaic([covered(300_000.0, 1), covered(300_640.0, 3)])

    assert laid.gs.geobox.shape == (32, 96)
    assert int(laid.red.values[0, 40]) == 0


def test_mosaic_lets_an_earlier_raster_win_an_overlap() -> None:
    laid = transform_grid.mosaic([covered(300_000.0, 1), covered(300_000.0, 9)])

    assert int(laid.red.values[0, 0]) == 1


def test_mosaic_lets_a_later_raster_fill_absent_pixels() -> None:
    holed = covered(300_000.0, 1)
    holed.red.values[4:8, 4:8] = 0

    laid = transform_grid.mosaic([holed, covered(300_000.0, 9)])

    assert int(laid.red.values[5, 5]) == 9
    assert int(laid.red.values[0, 0]) == 1


def test_mosaic_lets_a_later_raster_fill_absent_pixels_when_the_fill_is_nan() -> None:
    holed = covered_float(300_000.0, 1.0)
    holed.red.values[4:8, 4:8] = np.nan

    laid = transform_grid.mosaic([holed, covered_float(300_000.0, 9.0)])

    assert float(laid.red.values[5, 5]) == 9.0
    assert float(laid.red.values[0, 0]) == 1.0


def test_mosaic_keeps_the_dtype_and_the_laziness() -> None:
    import dask.array as da

    lazy = [
        raster(
            {"red": da.full((32, 32), value, dtype="uint16", chunks=(16, 16))},
            geobox((left, 5_000_000.0, left + 320.0, 5_000_320.0)),
        ).gs.write_nodata(0)
        for left, value in ((300_000.0, 1), (300_320.0, 2))
    ]

    laid = transform_grid.mosaic(lazy)

    assert laid.red.chunks is not None
    assert laid.red.dtype == np.dtype("uint16")


def test_mosaic_conforms_its_result() -> None:
    laid = transform_grid.mosaic([covered(300_000.0, 1), covered(300_320.0, 2)])

    assert laid.y.attrs["standard_name"] == "projection_y_coordinate"
    assert laid.red.attrs["_FillValue"] == 0


def test_mosaic_refuses_no_rasters() -> None:
    with pytest.raises(ValueError, match="at least one raster"):
        transform_grid.mosaic([])


def test_mosaic_refuses_a_raster_declaring_no_fill() -> None:
    with pytest.raises(ValueError, match="declares no fill value"):
        transform_grid.mosaic([covered(300_000.0, 1, fill=None)])


def test_mosaic_refuses_rasters_on_another_pixel_phase() -> None:
    # from_bbox snaps to the resolution, so the offset grid is built tight.
    shifted = GeoBox.from_bbox(
        (300_005.0, 5_000_005.0, 300_325.0, 5_000_325.0),
        crs=UTM,
        shape=(32, 32),
        tight=True,
    )
    other = raster(
        {"red": np.zeros(tuple(shifted.shape), "uint16")}, shifted
    ).gs.write_nodata(0)

    with pytest.raises(ValueError, match="not on one pixel grid"):
        transform_grid.mosaic([covered(300_000.0, 1), other])


def test_mosaic_refuses_rasters_on_different_crs() -> None:
    other_box = geobox(
        (300_000.0, 5_000_000.0, 300_320.0, 5_000_320.0), crs="EPSG:32634"
    )
    other = raster(
        {"red": np.zeros(tuple(other_box.shape), "uint16")}, other_box
    ).gs.write_nodata(0)

    with pytest.raises(ValueError, match=r"\['EPSG:32633', 'EPSG:32634'\]"):
        transform_grid.mosaic([covered(300_000.0, 1), other])


def test_mosaic_refuses_rasters_carrying_different_variables() -> None:
    other = covered(300_320.0, 2).rename_vars({"red": "nir"})

    with pytest.raises(ValueError, match="different variables"):
        transform_grid.mosaic([covered(300_000.0, 1), other])


def test_mosaic_refuses_rasters_on_different_dtypes() -> None:
    box = geobox((300_320.0, 5_000_000.0, 300_640.0, 5_000_320.0))
    other = raster({"red": np.zeros(tuple(box.shape), "float32")}, box, nodata=0)

    with pytest.raises(ValueError, match="different dtypes"):
        transform_grid.mosaic([covered(300_000.0, 1), other])


def test_mosaic_to_zarr_writes_what_the_in_memory_mosaic_holds(tmp_path) -> None:
    rasters = [covered(300_000.0, 1), covered(300_320.0, 2), covered(300_960.0, 3)]

    written = transform_grid.mosaic_to_zarr(rasters, tmp_path / "m.zarr")

    reopened = zarr.read(written)
    assert np.array_equal(
        reopened.red.values, transform_grid.mosaic(rasters).red.values
    )
    assert reopened.gs.geobox == transform_grid.mosaic(rasters).gs.geobox


def test_mosaic_to_zarr_writes_real_pixels_when_the_fill_is_nan(tmp_path) -> None:
    rasters = [covered_float(300_000.0, 1.0), covered_float(300_320.0, 2.0)]

    written = transform_grid.mosaic_to_zarr(rasters, tmp_path / "m.zarr")

    reopened = zarr.read(written)
    assert not np.isnan(reopened.red.values).all()
    assert float(reopened.red.values[0, 0]) == 1.0
    assert float(reopened.red.values[0, -1]) == 2.0


def test_mosaic_to_zarr_never_writes_a_chunk_for_a_gap(tmp_path) -> None:
    rasters = [covered(300_000.0, 1), covered(300_960.0, 3)]

    written = transform_grid.mosaic_to_zarr(rasters, tmp_path / "m.zarr")

    assert len(list(written.rglob("red/c/*/*"))) == len(rasters)


def test_mosaic_to_zarr_writes_a_readable_store(tmp_path) -> None:
    written = transform_grid.mosaic_to_zarr(
        [covered(300_000.0, 1), covered(300_320.0, 2)], tmp_path / "m.zarr"
    )

    assert zarr.read(written).gs.crs is not None


def test_mosaic_to_zarr_lays_rasters_opened_from_a_store(tmp_path) -> None:
    for name, (left, value) in {"a": (300_000.0, 1), "b": (300_320.0, 2)}.items():
        covered(left, value).gs.to_zarr(tmp_path / f"{name}.zarr")
    opened = [zarr.read(tmp_path / f"{name}.zarr", chunks={}) for name in ("a", "b")]

    written = transform_grid.mosaic_to_zarr(opened, tmp_path / "m.zarr")

    laid = zarr.read(written)
    assert int(laid.red.values[0, 0]) == 1
    assert int(laid.red.values[0, -1]) == 2


def test_mosaic_to_zarr_refuses_an_existing_destination(tmp_path) -> None:
    rasters = [covered(300_000.0, 1), covered(300_320.0, 2)]
    transform_grid.mosaic_to_zarr(rasters, tmp_path / "m.zarr")

    with pytest.raises(FileExistsError):
        transform_grid.mosaic_to_zarr(rasters, tmp_path / "m.zarr")

    assert transform_grid.mosaic_to_zarr(
        rasters, tmp_path / "m.zarr", overwrite=True
    ).exists()
