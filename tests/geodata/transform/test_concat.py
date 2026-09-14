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
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.errors import DroppedAttrsWarning, GeoSaveWarning
from geosave_engine.geodata.transform.concat import concat_time

UTM = "EPSG:32633"


def box(
    bbox: tuple[float, float, float, float] = (0, 0, 40, 40),
    *,
    resolution: float = 10,
    crs: str = UTM,
) -> GeoBox:
    """Build one small projected grid."""
    return GeoBox.from_bbox(bbox, crs=crs, resolution=resolution)


def series(
    *observed: str,
    grid: GeoBox | None = None,
    fill: int = 7,
    dtype: str = "uint16",
    nodata: int | None = 0,
    name: str = "red",
) -> xr.Dataset:
    """Build a raster observing `fill` at each label in `observed`."""
    grid = grid or box()
    labels = np.array(observed, "datetime64[ns]")
    pixels = np.stack([np.full(grid.shape, fill, dtype) for _ in labels])
    return build_raster({name: pixels}, grid, nodata=nodata, time=labels)


def dates(raster: xr.Dataset | xr.DataArray) -> list[str]:
    """Read a raster's time labels as `YYYY-MM-DD` strings."""
    return [str(label)[:10] for label in raster.coords["time"].values]


@contextmanager
def quiet() -> Iterator[None]:
    """Ignore warnings a join raises beside whatever is under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeoSaveWarning)
        yield


def test_rasters_lay_down_in_the_order_they_were_given() -> None:
    joined = concat_time([series("2024-01-05"), series("2024-02-10")])

    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_nothing_is_reordered_so_the_caller_may_sort_after() -> None:
    march, january = series("2024-03-01"), series("2024-01-05")

    joined = concat_time([march, january])

    assert dates(joined) == ["2024-03-01", "2024-01-05"]
    assert dates(joined.sortby("time")) == ["2024-01-05", "2024-03-01"]


def test_overlapping_and_repeated_labels_lay_down_untouched() -> None:
    joined = concat_time(
        [series("2024-01-01", "2024-02-01"), series("2024-01-15", "2024-01-01")]
    )

    assert dates(joined) == ["2024-01-01", "2024-02-01", "2024-01-15", "2024-01-01"]


def test_an_unlabelled_or_empty_axis_lays_down_all_the_same() -> None:
    grid = box()
    empty = build_raster(
        {"red": np.zeros((0, *grid.shape), "uint16")},
        grid,
        nodata=0,
        time=np.array([], "datetime64[ns]"),
    )

    joined = concat_time([empty, series("2024-01-05")])

    assert dates(joined) == ["2024-01-05"]


def test_shared_attrs_survive_and_conflicting_ones_are_dropped() -> None:
    first, second = series("2024-01-05"), series("2024-02-10")
    for raster in (first, second):
        raster.attrs.update({"title": "scene", "license": "CC0"})
    second.attrs["title"] = "other"

    with pytest.warns(DroppedAttrsWarning):
        joined = concat_time([first, second])

    assert joined.attrs["license"] == "CC0"
    assert "title" not in joined.attrs


def test_the_dtype_and_laziness_of_the_rasters_survive() -> None:
    lazy = concat_time(
        [series("2024-01-05").chunk(), series("2024-02-10").chunk()],
    )

    assert lazy.red.dtype == np.dtype("uint16")
    assert lazy.red.chunks is not None


def test_a_looser_join_pads_what_a_raster_does_not_reach() -> None:
    west = series("2024-01-05", fill=7)
    east = series("2024-02-10", grid=box((20, 0, 60, 40)), fill=9)

    with quiet():
        joined = concat_time([west, east], join="outer")

    assert joined.red.shape == (2, 4, 6)
    assert joined.red.dtype == np.dtype("uint16")
    assert list(joined.red.isel(time=0, y=0).values) == [7, 7, 7, 7, 0, 0]
    assert list(joined.red.isel(time=1, y=0).values) == [0, 0, 9, 9, 9, 9]


def test_an_exact_join_refuses_rasters_of_different_extent() -> None:
    with quiet(), pytest.raises(xr.AlignmentError):
        concat_time(
            [series("2024-01-05"), series("2024-02-10", grid=box((20, 0, 60, 40)))]
        )


def test_padding_refuses_a_variable_carrying_no_fill_value() -> None:
    with pytest.raises(ValueError, match="carries no fill value"):
        concat_time(
            [
                series("2024-01-05", nodata=None),
                series("2024-02-10", grid=box((20, 0, 60, 40)), nodata=None),
            ],
            join="outer",
        )


def test_a_stack_concatenates_group_by_group() -> None:
    first = stack({"s2": series("2024-01-05"), "s1": series("2024-01-05", name="vv")})
    second = stack({"s2": series("2024-02-10"), "s1": series("2024-02-10", name="vv")})

    joined = concat_time([first, second])

    assert joined.gs.groups == ("s2", "s1")
    assert dates(joined["s2"]) == ["2024-01-05", "2024-02-10"]
    assert dates(joined["s1"]) == ["2024-01-05", "2024-02-10"]


def test_stacks_naming_different_groups_are_refused() -> None:
    with pytest.raises(ValueError, match="name different rasters"):
        concat_time(
            [stack({"a": series("2024-01-05")}), stack({"b": series("2024-02-10")})]
        )


def test_concatenating_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="pass at least one"):
        concat_time([])


def test_a_dataarray_concatenates_as_a_dataarray() -> None:
    joined = concat_time([series("2024-01-05").red, series("2024-02-10").red])

    assert isinstance(joined, xr.DataArray)
    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_ungeoreferenced_rasters_concatenate_in_pixel_space() -> None:
    def bare(label: str) -> xr.Dataset:
        return xr.Dataset(
            {"red": (("time", "y", "x"), np.zeros((1, 4, 4), "uint16"))},
            coords={"time": np.array([label], "datetime64[ns]")},
        )

    joined = concat_time([bare("2024-01-05"), bare("2024-02-10")])

    assert joined.red.shape == (2, 4, 4)
    assert joined.gs.geobox is None


def test_mixing_placed_and_unplaced_rasters_is_refused() -> None:
    bare = xr.Dataset(
        {"red": (("time", "y", "x"), np.zeros((1, 4, 4), "uint16"))},
        coords={"time": np.array(["2024-02-10"], "datetime64[ns]")},
    )

    with pytest.raises(ValueError, match="EPSG:32633"):
        concat_time([series("2024-01-05"), bare])


@pytest.mark.parametrize(
    ("other", "mismatch"),
    [
        (box(resolution=20), "not equal along these coord"),
        (box(crs="EPSG:32748"), "EPSG:32748"),
        (
            GeoBox((4, 4), Affine(10.0, 0.0, 3.0, 0.0, -10.0, 40.0), CRS(UTM)),
            "not equal along these coord",
        ),
    ],
    ids=["resolution", "crs", "pixel phase"],
)
def test_rasters_no_one_grid_holds_are_refused(other: GeoBox, mismatch: str) -> None:
    with pytest.raises(ValueError, match=mismatch):
        concat_time([series("2024-01-05"), series("2024-02-10", grid=other)])


def test_grids_covering_different_ground_are_refused() -> None:
    # Laying these end to end would interleave x labels rather than extend time.
    with pytest.raises(ValueError, match="not equal along these coord"):
        concat_time(
            [series("2024-01-05"), series("2024-02-10", grid=box((40, 0, 80, 40)))]
        )


def test_rasters_placed_by_nothing_still_lay_end_to_end() -> None:
    bare = series("2024-01-05").drop_vars(["y", "x", "spatial_ref"])
    later = series("2024-02-10").drop_vars(["y", "x", "spatial_ref"])

    assert dates(concat_time([bare, later])) == ["2024-01-05", "2024-02-10"]


def test_an_origin_a_hair_off_a_pixel_still_lays_end_to_end() -> None:
    # snap_to and reprojection both land an origin just below zero, which the
    # pixel labels absorb.
    drifted = GeoBox((4, 4), Affine(10.0, 0.0, -4.44e-16, 0.0, -10.0, 40.0), CRS(UTM))

    joined = concat_time([series("2024-01-05"), series("2024-02-10", grid=drifted)])

    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_a_grid_lays_end_to_end_with_itself_however_it_is_oriented() -> None:
    south_up = GeoBox((4, 4), Affine(10.0, 0.0, 0.0, 0.0, 10.0, 0.0), CRS(UTM))

    joined = concat_time([series("2024-01-05", grid=south_up)] * 2)

    assert dates(joined) == ["2024-01-05", "2024-01-05"]
    with pytest.raises(ValueError, match="not equal along these coord"):
        concat_time([series("2024-01-05"), series("2024-02-10", grid=south_up)])
