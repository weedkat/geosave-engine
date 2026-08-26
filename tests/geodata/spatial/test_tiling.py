"""Cutting a surface into tiles and merging them back.

The round trip is the contract: `tiles()` then a `GeoStitcher` must return
the original pixels exactly, including the padded and overlapping cases.
Every validation here guards a merge that would otherwise produce a wrong
surface silently rather than raising.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from geosave_engine.geodata.spatial import GeoRaster, GeoStitcher, GeoTile, GeoVector

from .conftest import is_lazy, make_anchor, make_lazy_raster, make_raster, make_vector


def cut(raster: GeoRaster, **kwargs) -> list:
    return list(raster.tiles(**kwargs))


def merge(*tiles: GeoTile, **kwargs) -> list[GeoRaster]:
    """Every surface `tiles` complete, through one stitcher."""
    stitcher = GeoStitcher()
    stitcher.add(*tiles)
    return list(stitcher.flush(**kwargs))


class TestRoundTrip:
    def test_evenly_divisible_surface(self):
        raster = make_raster(width=64, height=64)
        merged = merge(*cut(raster, tile_size_px=32))
        assert len(merged) == 1
        np.testing.assert_array_equal(merged[0].data.values, raster.data.values)

    def test_padded_surface_comes_back_at_its_original_size(self):
        raster = make_raster(width=50, height=50)
        merged = merge(*cut(raster, tile_size_px=32))
        assert merged[0].anchor.shape == (50, 50)
        np.testing.assert_array_equal(merged[0].data.values, raster.data.values)

    def test_four_dimensional_surface(self):
        times = [datetime(2024, 1, 1), datetime(2024, 2, 1)]
        raster = make_raster(times=times, dtype="float32", nodata=np.nan, time=("2024-01-01", "2024-02-01"))
        merged = merge(*cut(raster, tile_size_px=32))
        assert merged[0].data.dims == ("time", "band", "y", "x")
        np.testing.assert_allclose(merged[0].data.values, raster.data.values)

    def test_overlapping_tiles_merge_through_a_window(self):
        raster = make_raster(bands=("B04",), dtype="float32", nodata=np.nan)
        stitcher = GeoStitcher(window="hann")
        stitcher.add(*cut(raster, tile_size_px=32, overlap=8))
        merged = list(stitcher.flush())
        assert len(merged) == 1 and merged[0].anchor.shape == (64, 64)

    def test_merged_surface_carries_no_tiling_stamp(self):
        raster = make_raster()
        merged = merge(*cut(raster, tile_size_px=32))[0]
        assert merged.tiling is None

    def test_prediction_reanchored_through_the_tile_still_merges(self):
        raster = make_raster()
        predictions = [
            tile.anchor.to_geotile(np.ones((32, 32), dtype="uint8"), bands=("class",), header=tile.header)
            for tile in cut(raster, tile_size_px=32)
        ]
        merged = merge(*predictions)[0]
        assert merged.bands == ("class",)
        assert merged.anchor.shape == (64, 64)


class TestTileGeometry:
    def test_every_tile_shares_one_group(self):
        tiles = cut(make_raster(), tile_size_px=32)
        assert len({tile.tiling.group_id for tile in tiles}) == 1

    def test_tile_ids_are_row_major_and_complete(self):
        tiles = cut(make_raster(), tile_size_px=32)
        assert [tile.tiling.tile_id for tile in tiles] == [0, 1, 2, 3]

    def test_tiles_do_not_read_pixels(self):
        assert is_lazy(next(iter(make_lazy_raster().tiles(tile_size_px=32))).to_raster())

    def test_padded_tiles_do_not_read_pixels_either(self):
        assert is_lazy(next(iter(make_lazy_raster(width=50, height=50).tiles(tile_size_px=32))).to_raster())

    def test_constant_mode_needs_a_declared_nodata(self):
        raster = make_raster(width=50, height=50, nodata=None)
        with pytest.raises(ValueError, match="rebase\\(nodata="):
            cut(raster, tile_size_px=32, mode="constant")

    def test_stride_wider_than_the_tile_is_rejected(self):
        with pytest.raises(ValueError, match="stride_px must be in"):
            cut(make_raster(), tile_size_px=32, stride_px=64)


class TestVectorFanOut:
    def test_features_reach_the_tiles_they_touch(self):
        raster = make_raster(vector=make_vector())
        carried = [tile for tile in cut(raster, tile_size_px=32) if tile.vector is not None]
        assert carried, "no tile carried the vector"

    def test_a_feature_split_across_tiles_collapses_back_to_one_row(self):
        raster = make_raster(vector=make_vector(size=400))
        merged = merge(*cut(raster, tile_size_px=32))[0]
        assert merged.vector is not None
        assert len(merged.vector) == 1

    def test_vector_false_yields_bare_tiles(self):
        raster = make_raster(vector=make_vector())
        assert all(tile.vector is None for tile in cut(raster, tile_size_px=32, vector=False))


class TestStitcherValidation:
    def test_a_tile_without_a_stamp_is_rejected(self):
        loose = make_anchor(width=32, height=32).to_geotile(
            np.ones((1, 32, 32), dtype="uint16"), bands=["B04"]
        )
        with pytest.raises(ValueError, match="carries no tiling stamp"):
            GeoStitcher().add(loose)

    def test_the_same_tile_twice_is_rejected(self):
        tiles = cut(make_raster(), tile_size_px=32)
        stitcher = GeoStitcher()
        stitcher.add(tiles[0])
        with pytest.raises(ValueError, match="already added"):
            stitcher.add(tiles[0])

    def test_a_tile_with_a_different_dtype_is_rejected(self):
        tiles = cut(make_raster(), tile_size_px=32)
        stitcher = GeoStitcher()
        stitcher.add(tiles[0])
        with pytest.raises(ValueError, match="dtype"):
            stitcher.add(tiles[1].astype("int32", nodata=-1))

    def test_an_incomplete_group_is_rejected_by_default(self):
        tiles = cut(make_raster(), tile_size_px=32)
        with pytest.raises(ValueError, match="missing"):
            merge(*tiles[:2])

    def test_validation_happens_before_iteration(self):
        """flush() must raise from the call, not from consuming its result."""
        tiles = cut(make_raster(), tile_size_px=32)
        stitcher = GeoStitcher()
        stitcher.add(*tiles[:2])
        with pytest.raises(ValueError):
            stitcher.flush()  # not wrapped in list() on purpose

    def test_partial_merge_fills_holes_with_the_declared_nodata(self):
        tiles = cut(make_raster(nodata=9999), tile_size_px=32)
        stitcher = GeoStitcher()
        stitcher.add(*tiles[:2])
        merged = next(iter(stitcher.flush(allow_partial=True)))
        assert 9999 in np.unique(merged.data.values)

    def test_partial_merge_without_a_sentinel_is_rejected(self):
        """Without nodata the holes would merge as zeros and read as real pixels."""
        tiles = cut(make_raster(nodata=None), tile_size_px=32)
        stitcher = GeoStitcher()
        stitcher.add(*tiles[:2])
        with pytest.raises(ValueError, match="no nodata"):
            list(stitcher.flush(allow_partial=True))

    def test_drain_releases_only_completed_groups(self):
        tiles = cut(make_raster(), tile_size_px=32)
        stitcher = GeoStitcher()
        stitcher.add(*tiles[:2])
        assert list(stitcher.drain()) == []
        stitcher.add(*tiles[2:])
        assert len(list(stitcher.drain())) == 1
        assert len(stitcher) == 0

    def test_identical_cuts_share_one_group_id(self):
        """group_id names the cut, not the run, so a resumed or distributed run merges."""
        first = cut(make_raster(), tile_size_px=32)
        second = cut(make_raster(), tile_size_px=32)
        assert first[0].tiling.group_id == second[0].tiling.group_id

    def test_a_name_separates_two_otherwise_identical_cuts(self):
        merged = merge(
            *cut(make_raster(), tile_size_px=32, name="a"),
            *cut(make_raster(), tile_size_px=32, name="b"),
        )
        assert len(merged) == 2

    def test_cuts_over_different_time_spans_stay_separate(self):
        early = make_raster(time=("2024-01-01", "2024-01-31"))
        late = make_raster(time=("2024-02-01", "2024-02-29"))
        merged = merge(*cut(early, tile_size_px=32), *cut(late, tile_size_px=32))
        assert len(merged) == 2


class TestCrop:
    def test_crop_returns_a_surface_on_the_requested_window(self):
        raster = make_raster()
        window = make_anchor(width=16, height=16).geobox
        cropped = raster.crop(window)
        assert isinstance(cropped, GeoRaster)
        assert cropped.anchor.shape == (16, 16)
        assert cropped.tiling is None

    def test_crop_accepts_a_dated_anchors_geobox(self):
        """A window anchor normally carries a time span; only its grid is read."""
        from geosave_engine.geodata.spatial import GeoAnchor

        raster = make_raster()
        window = GeoAnchor.from_bbox(
            (500_000, 5_000_000, 500_160, 5_000_160), resolution=10, crs="EPSG:32633", timespan="2024-01-15"
        )
        assert raster.crop(window.geobox).anchor.shape == (16, 16)

    def test_crop_rejects_an_off_grid_window(self):
        off_grid = make_anchor(width=8, height=8, x0=500_005).geobox
        with pytest.raises(ValueError, match="reproject"):
            make_raster().crop(off_grid)

    def test_crop_rejects_a_window_reaching_outside(self):
        too_big = make_anchor(width=128, height=128).geobox
        with pytest.raises(ValueError, match="fully inside"):
            make_raster().crop(too_big)

    def test_crop_filters_the_vector(self):
        raster = make_raster(vector=make_vector(x0=500_500, y0=5_000_500, size=50))
        far_corner = make_anchor(width=8, height=8).geobox
        assert raster.crop(far_corner).vector is None

    def test_crop_stays_lazy(self):
        window = make_anchor(width=16, height=16).geobox
        assert is_lazy(make_lazy_raster().crop(window))


class TestVectorConcat:
    def test_identical_features_deduplicate(self):
        one = make_vector()
        assert len(GeoVector.concat(one, make_vector())) == 1

    def test_distinct_features_all_survive(self):
        combined = GeoVector.concat(make_vector(), make_vector(x0=500_400, y0=5_000_400))
        assert len(combined) == 2

    def test_all_none_gives_none(self):
        assert GeoVector.concat(None, None) is None
