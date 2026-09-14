from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.transform.tiling import Tiles


def _raster(
    height: int, width: int, fill: float = 0.0, seed: int | None = None
) -> xr.Dataset:
    grid = GeoBox.from_bbox(
        (0, 0, width * 10, height * 10), "EPSG:32748", resolution=10
    )
    pixels = (
        np.random.default_rng(seed).integers(0, 5000, (height, width)).astype("float32")
        if seed is not None
        else np.full((height, width), fill, "float32")
    )
    return xr.Dataset({"B": (("y", "x"), pixels)}, coords=xr_coords(grid))


def _feed(tiles: Tiles, merger, order=None) -> None:
    numbers = range(len(tiles)) if order is None else order
    merger.add({int(n): np.asarray(tiles[int(n)]["B"].values) for n in numbers})


@pytest.mark.parametrize("overlap", [0, 32, 64, 17])
def test_merge_rebuilds_every_raster_exactly(overlap: int) -> None:
    rasters = [_raster(600, 600, seed=1), _raster(512, 300, seed=2)]
    tiles = Tiles(rasters, (256, 256), overlap=overlap)
    merger = tiles.merger()

    _feed(tiles, merger, np.random.default_rng(0).permutation(len(tiles)))

    rebuilt = merger.merge()
    for ordinal, source in enumerate(rasters):
        assert np.array_equal(rebuilt[ordinal].values, source.B.values)
        assert rebuilt[ordinal].odc.geobox == source.odc.geobox


@pytest.mark.parametrize("window", ["hann", "overlap-tile", "blackmanharris"])
def test_window_leaves_no_blank_border(window: str) -> None:
    source = _raster(600, 600, fill=7.0)
    tiles = Tiles([source], (256, 256), overlap=32)
    merger = tiles.merger(window=window)

    _feed(tiles, merger)

    rebuilt = merger.merge()[0].values
    assert rebuilt.shape == (600, 600)
    np.testing.assert_allclose(rebuilt, 7.0, atol=1e-4)


def test_tile_keeps_shape_and_grid_past_the_trailing_edge() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256), overlap=32)

    for index in (0, len(tiles) - 1):
        tile = tiles[index]
        assert dict(tile.sizes) == {"y": 256, "x": 256}
        assert tile.odc.geobox.crs == tiles[0].odc.geobox.crs


def test_locate_numbers_rasters_in_cutting_order() -> None:
    tiles = Tiles([_raster(600, 600), _raster(512, 300)], (256, 256), overlap=32)

    assert tiles.locate(0)[0] == 0
    assert tiles.locate(len(tiles) - 1)[0] == 1
    assert tiles.locate(-1) == tiles.locate(len(tiles) - 1)
    with pytest.raises(IndexError):
        tiles.locate(len(tiles))


def test_merge_drains_finished_rasters_and_leaves_the_rest() -> None:
    rasters = [_raster(600, 600, seed=3), _raster(512, 300, seed=4)]
    tiles = Tiles(rasters, (256, 256), overlap=32)
    merger = tiles.merger()
    first = [n for n in range(len(tiles)) if tiles.locate(n)[0] == 0]

    _feed(tiles, merger, first[:-1])
    assert merger.merge() == {}
    assert merger.pending == {0: 1}

    _feed(tiles, merger, first[-1:])
    assert sorted(merger.merge()) == [0]
    assert merger.pending == {}


def test_results_of_several_rasters_may_arrive_in_one_batch() -> None:
    rasters = [_raster(600, 600, seed=5), _raster(512, 300, seed=6)]
    tiles = Tiles(rasters, (256, 256), overlap=32)
    merger = tiles.merger()

    order = np.random.default_rng(1).permutation(len(tiles))
    for start in range(0, len(order), 8):
        _feed(tiles, merger, order[start : start + 8])

    rebuilt = merger.merge()
    assert np.array_equal(rebuilt[0].values, rasters[0].B.values)
    assert np.array_equal(rebuilt[1].values, rasters[1].B.values)


def test_bands_ride_a_leading_axis_through_the_merge() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256), overlap=32)
    merger = tiles.merger()

    merger.add(
        {index: np.stack([tiles[index]["B"].values] * 3) for index in range(len(tiles))}
    )

    rebuilt = merger.merge()[0]
    assert rebuilt.dims == ("band", "y", "x")
    assert rebuilt.shape == (3, 600, 600)


def test_a_stack_is_cut_group_by_group() -> None:
    grouped = stack({"optical": _raster(600, 600, 7.0), "dem": _raster(600, 600, 3.0)})
    tiles = Tiles([grouped], (256, 256), overlap=32)

    tile = tiles[len(tiles) - 1]
    assert isinstance(tile, xr.DataTree)
    assert tile.gs.groups == grouped.gs.groups
    assert tile.gs.geobox.shape == (256, 256)


def test_a_dataarray_is_cut_as_a_dataarray() -> None:
    tiles = Tiles([_raster(600, 600).B], (256, 256), overlap=32)

    assert isinstance(tiles[0], xr.DataArray)
    assert dict(tiles[0].sizes) == {"y": 256, "x": 256}


def test_leading_axes_ride_along_untouched() -> None:
    grid = GeoBox.from_bbox((0, 0, 6000, 6000), "EPSG:32748", resolution=10)
    source = xr.Dataset(
        {"B": (("time", "band", "y", "x"), np.zeros((4, 6, 600, 600), "float32"))},
        coords={**xr_coords(grid), "time": np.arange(4), "band": list("abcdef")},
    )
    tiles = Tiles([source], (256, 256), overlap=32)

    assert dict(tiles[len(tiles) - 1].sizes) == {
        "time": 4,
        "band": 6,
        "y": 256,
        "x": 256,
    }


def test_a_tile_stays_lazy_while_its_raster_is() -> None:
    pytest.importorskip("dask")
    tiles = Tiles(
        [_raster(600, 600).chunk({"y": 256, "x": 256})], (256, 256), overlap=32
    )

    assert hasattr(tiles[5].B.data, "compute")


def test_a_cut_refuses_what_it_cannot_tile() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Tiles([], (256, 256))
    with pytest.raises(ValueError, match="larger than raster 0"):
        Tiles([_raster(600, 600)], (1024, 1024))
    with pytest.raises(ValueError, match="not a padding kind"):
        Tiles([_raster(600, 600)], (256, 256), mode="bogus")  # type: ignore[arg-type]


def test_a_raster_with_no_geobox_is_cut_in_pixel_space() -> None:
    source = xr.Dataset({"B": (("y", "x"), np.full((600, 600), 7.0, "float32"))})
    tiles = Tiles([source], (256, 256), overlap=32)
    merger = tiles.merger()

    _feed(tiles, merger)

    tile = tiles[0]
    assert "spatial_ref" not in tile.coords
    rebuilt = merger.merge()[0]
    np.testing.assert_allclose(rebuilt.values, 7.0, atol=1e-4)


def test_a_merger_refuses_a_result_with_a_stray_leading_axis() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256), overlap=32)
    merger = tiles.merger()

    with pytest.raises(ValueError, match=r"tile 0's result carries 2 leading axes"):
        merger.add({0: np.zeros((2, 3, 256, 256), "float32")})


def test_a_merger_with_named_leading_dims_round_trips_them() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256), overlap=32)
    merger = tiles.merger(leading_dims=("time", "band"))

    for index in range(len(tiles)):
        tile = np.asarray(tiles[index]["B"].values)
        prediction = np.stack(
            [np.stack([tile, tile, tile])] * 2
        )  # (time=2, band=3, y, x)
        merger.add({index: prediction})

    rebuilt = merger.merge()[0]
    assert rebuilt.dims == ("time", "band", "y", "x")
    assert rebuilt.shape == (2, 3, 600, 600)


def test_a_merger_refuses_leading_dims_arity_mismatch() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256), overlap=32)
    merger = tiles.merger(leading_dims=("time", "band"))

    with pytest.raises(ValueError, match="leading_dims names"):
        merger.add({0: np.zeros((3, 256, 256), "float32")})


def test_a_merger_refuses_a_raster_whose_leading_shape_drifts() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256), overlap=32)
    merger = tiles.merger()

    merger.add({0: np.zeros((3, 256, 256), "float32")})
    with pytest.raises(ValueError, match="started with"):
        merger.add({1: np.zeros((4, 256, 256), "float32")})


def test_a_window_needs_tiles_that_overlap() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256))

    with pytest.raises(ValueError, match="needs tiles cut with an overlap"):
        tiles.merger(window="hann")
