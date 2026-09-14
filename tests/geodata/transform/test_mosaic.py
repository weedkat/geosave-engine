from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.geom import CRS

from geosave_engine.geodata.core.raster import raster as build_raster
from geosave_engine.geodata.errors import GeoSaveWarning
from geosave_engine.geodata.transform.composite import mosaic

UTM = "EPSG:32633"


def box(
    bbox: tuple[float, float, float, float] = (0, 0, 40, 40),
    *,
    resolution: float = 10,
    crs: str = UTM,
) -> GeoBox:
    """Build one small projected grid."""
    return GeoBox.from_bbox(bbox, crs=crs, resolution=resolution)


def granule(
    grid: GeoBox | None = None,
    *,
    fill: int = 7,
    dtype: str = "uint16",
    nodata: int | None = 0,
    name: str = "red",
) -> xr.Dataset:
    """Build a timeless raster holding `fill` at every pixel."""
    grid = grid or box()
    return build_raster({name: np.full(grid.shape, fill, dtype)}, grid, nodata=nodata)


@contextmanager
def quiet() -> Iterator[None]:
    """Ignore warnings a mosaic raises beside whatever is under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeoSaveWarning)
        yield


def row(raster: xr.Dataset, name: str = "red") -> list[int]:
    """Read a raster's top row as plain integers."""
    return [int(value) for value in raster[name].isel(y=0).values]


def test_rasters_lay_onto_the_ground_they_jointly_cover() -> None:
    with quiet():
        laid = mosaic([granule(), granule(box((20, 0, 60, 40)), fill=9)])

    assert laid.red.shape == (4, 6)
    assert laid.odc.geobox.extent.boundingbox[0] == 0.0
    assert laid.odc.geobox.extent.boundingbox[2] == 60.0


def test_the_first_raster_wins_an_overlapping_pixel() -> None:
    with quiet():
        laid = mosaic([granule(fill=7), granule(box((20, 0, 60, 40)), fill=9)])

    assert row(laid) == [7, 7, 7, 7, 9, 9]


def test_the_last_raster_wins_when_it_is_asked_to() -> None:
    with quiet():
        laid = mosaic(
            [granule(fill=7), granule(box((20, 0, 60, 40)), fill=9)], method="last"
        )

    assert row(laid) == [7, 7, 9, 9, 9, 9]


def test_ground_no_raster_covers_holds_the_fill_value() -> None:
    with quiet():
        laid = mosaic([granule(fill=7), granule(box((80, 0, 120, 40)), fill=9)])

    assert laid.red.shape == (4, 12)
    assert row(laid) == [7, 7, 7, 7, 0, 0, 0, 0, 9, 9, 9, 9]


def test_a_nodata_pixel_is_filled_by_a_later_raster() -> None:
    covered = granule(fill=7)
    covered["red"][0, 0] = 0

    with quiet():
        laid = mosaic([covered, granule(fill=9)])

    assert row(laid) == [9, 7, 7, 7]


def test_three_rasters_lay_down_in_preference_order() -> None:
    with quiet():
        laid = mosaic(
            [
                granule(fill=7),
                granule(box((20, 0, 60, 40)), fill=8),
                granule(box((40, 0, 80, 40)), fill=9),
            ]
        )

    assert row(laid) == [7, 7, 7, 7, 8, 8, 9, 9]


def test_one_raster_mosaics_to_itself() -> None:
    laid = mosaic([granule(fill=7)])

    assert laid.red.shape == (4, 4)
    assert row(laid) == [7, 7, 7, 7]


def test_the_dtype_and_laziness_of_the_rasters_survive() -> None:
    with quiet():
        laid = mosaic([granule().chunk(), granule(box((20, 0, 60, 40))).chunk()])

    assert laid.red.dtype == np.dtype("uint16")
    assert laid.red.chunks is not None


def test_the_result_carries_the_crs_it_was_laid_on() -> None:
    with quiet():
        laid = mosaic([granule(), granule(box((20, 0, 60, 40)))])

    assert laid.odc.crs is not None
    assert laid.odc.crs.epsg == 32633


def test_mosaicking_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="pass at least one"):
        mosaic([])


def test_a_raster_carrying_no_fill_value_is_refused() -> None:
    with quiet(), pytest.raises(ValueError, match="carries no fill value"):
        mosaic([granule(nodata=None), granule(box((20, 0, 60, 40)), nodata=None)])


def test_rasters_naming_different_variables_are_refused() -> None:
    with pytest.raises(ValueError, match="are not in the raster's variables"):
        mosaic([granule(), granule(name="nir")])


def test_rasters_carrying_no_grid_are_refused() -> None:
    bare = xr.Dataset({"red": (("y", "x"), np.zeros((4, 4), "uint16"))})
    bare["red"].attrs["_FillValue"] = 0

    with pytest.raises(ValueError, match="sits on no locatable grid"):
        mosaic([bare, bare])


@pytest.mark.parametrize(
    ("other", "mismatch"),
    [
        (box(resolution=20), "resolution"),
        (box(crs="EPSG:32748"), "CRS EPSG:32748 against EPSG:32633"),
        (GeoBox((4, 4), Affine(10.0, 0.0, 3.0, 0.0, -10.0, 40.0), CRS(UTM)), "phase"),
    ],
    ids=["resolution", "crs", "pixel phase"],
)
def test_rasters_no_one_grid_holds_are_refused(other: GeoBox, mismatch: str) -> None:
    with quiet(), pytest.raises(ValueError, match=mismatch):
        mosaic([granule(), granule(other, fill=9)])
