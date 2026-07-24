"""Unit tests for StacSource: temporal patching/reducing/stacking, all per-source now.

No network — odc_load is monkeypatched to a synthetic in-memory Dataset;
SearchClient is a fake returning a fixed item list.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.stac.source import StacSource
from geosave_engine.geodata.tile import GeoAnchor

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _item(item_id: str, when: datetime) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry={"type": "Point", "coordinates": [0.0, 0.0]},
        bbox=[0.0, 0.0, 0.0, 0.0],
        datetime=when,
        properties={},
    )


class _FakeClient:
    """Fake SearchClient — returns a fixed item list, ignores the query."""

    def __init__(self, items: list[pystac.Item]) -> None:
        self.items = items

    def search(self, query):
        return list(self.items)


def _fake_odc_load(items, *, geobox, bands, resampling, chunks, dtype, groupby):
    """Synthetic odc-stac-shaped Dataset: one time step per item, all-valid pixels.

    Real odc_load strips tzinfo before building the time coord (see
    odc.stac._stac_load._extract_timestamps) — matched here so this fake
    behaves like the real thing, not like a naive-vs-aware mismatch that
    only this synthetic path would ever hit.
    """
    times = [item.datetime.replace(tzinfo=None) for item in items]
    coords = dict(xr_coords(geobox))
    coords["time"] = times
    data_vars = {
        band: (("time", "y", "x"), np.ones((len(times), geobox.height, geobox.width), dtype=dtype))
        for band in bands
    }
    return xr.Dataset(data_vars, coords=coords)


def _anchor(start: datetime, end: datetime) -> GeoAnchor:
    gb = GeoBox.from_bbox(BBOX, crs=UTM, resolution=10, anchor="edge")
    return GeoAnchor(geobox=gb, datetime=(start, end))


@pytest.fixture(autouse=True)
def _patch_odc_load(monkeypatch):
    monkeypatch.setattr("geosave_engine.geodata.stac.source.odc_load", _fake_odc_load)


class TestSceneGranularity:
    def test_one_tile_per_scene(self):
        items = [
            _item("a", datetime(2023, 1, 5, tzinfo=timezone.utc)),
            _item("b", datetime(2023, 2, 12, tzinfo=timezone.utc)),
            _item("c", datetime(2023, 3, 20, tzinfo=timezone.utc)),
        ]
        source = StacSource(_FakeClient(items), collection="test-col", bands=["B02"])
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 12, 31))

        tiles = list(source.load(anchor))

        assert len(tiles) == 3
        for tile in tiles:
            assert tuple(tile.data.shape) == (1, 1, 32, 32)  # time, band, y, x

    def test_sub_second_scene_timestamp_matches_its_own_window(self):
        """Real STAC acquisition times carry microseconds — a point window built
        from a second-truncated timestamp must still exact-match the full-
        precision original, or every real scene misses its own bucket."""
        items = [_item("a", datetime(2023, 1, 5, 10, 15, 23, 456000, tzinfo=timezone.utc))]
        source = StacSource(_FakeClient(items), collection="test-col", bands=["B02"])
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 12, 31))

        tiles = list(source.load(anchor))

        assert len(tiles) == 1
        assert tuple(tiles[0].data.shape) == (1, 1, 32, 32)

    def test_temporal_slots_groups_consecutive_scenes_drops_trailing(self):
        items = [_item(str(i), datetime(2023, 1, 1 + i, tzinfo=timezone.utc)) for i in range(5)]
        source = StacSource(
            _FakeClient(items), collection="test-col", bands=["B02"], temporal_slots=2
        )
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 12, 31))

        tiles = list(source.load(anchor))

        assert len(tiles) == 2  # 5 scenes / slots=2, strides=2 (default) -> 2 full groups, trailing 1 dropped
        for tile in tiles:
            assert tuple(tile.data.shape) == (2, 1, 32, 32)

    def test_temporal_strides_produces_overlapping_windows(self):
        items = [_item(str(i), datetime(2023, 1, 1 + i, tzinfo=timezone.utc)) for i in range(5)]
        source = StacSource(
            _FakeClient(items),
            collection="test-col",
            bands=["B02"],
            temporal_slots=3,
            temporal_strides=1,
        )
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 12, 31))

        tiles = list(source.load(anchor))

        assert len(tiles) == 3  # windows start at scene 0, 1, 2 (stride 1) -> [0:3], [1:4], [2:5]
        for tile in tiles:
            assert tuple(tile.data.shape) == (3, 1, 32, 32)



class TestCalendarGranularity:
    def test_month_reduces_multiple_scenes_to_one_step(self):
        items = [
            _item("a", datetime(2023, 1, 5, tzinfo=timezone.utc)),
            _item("b", datetime(2023, 1, 20, tzinfo=timezone.utc)),
        ]
        source = StacSource(
            _FakeClient(items),
            collection="test-col",
            bands=["B02"],
            temporal_granularity="month",
            temporal_reduce="median",
        )
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 1, 31))

        tiles = list(source.load(anchor))

        assert len(tiles) == 1
        assert tuple(tiles[0].data.shape) == (1, 1, 32, 32)

    def test_empty_month_dropped_without_fallback(self):
        items = [_item("a", datetime(2023, 1, 15, tzinfo=timezone.utc))]
        source = StacSource(
            _FakeClient(items), collection="test-col", bands=["B02"], temporal_granularity="month"
        )
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 2, 28))

        tiles = list(source.load(anchor))

        assert len(tiles) == 1  # february bucket had nothing, dropped

    def test_empty_month_uses_fallback(self):
        items = [_item("a", datetime(2023, 1, 15, tzinfo=timezone.utc))]
        source = StacSource(
            _FakeClient(items),
            collection="test-col",
            bands=["B02"],
            temporal_granularity="month",
            temporal_fallback=True,
        )
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 2, 28))

        tiles = list(source.load(anchor))

        assert len(tiles) == 2  # february bucket substitutes january's only scene


class TestErrors:
    def test_no_items_raises_lazily(self):
        source = StacSource(_FakeClient([]), collection="test-col", bands=["B02"])
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 1, 31))

        gen = source.load(anchor)  # generator creation itself must not raise
        with pytest.raises(AnchorFetchError):
            next(gen)


class TestLazyLoad:
    def test_lazy_load_true_skips_download(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "geosave_engine.geodata.stac.source.download",
            lambda tile, **kwargs: calls.append(tile) or tile,
        )
        items = [_item("a", datetime(2023, 1, 5, tzinfo=timezone.utc))]
        source = StacSource(_FakeClient(items), collection="test-col", bands=["B02"])
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 12, 31))

        tiles = list(source.load(anchor, lazy_load=True))

        assert len(tiles) == 1
        assert calls == []

    def test_lazy_load_false_calls_download(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "geosave_engine.geodata.stac.source.download",
            lambda tile, **kwargs: calls.append(tile) or tile,
        )
        items = [_item("a", datetime(2023, 1, 5, tzinfo=timezone.utc))]
        source = StacSource(_FakeClient(items), collection="test-col", bands=["B02"])
        anchor = _anchor(datetime(2023, 1, 1), datetime(2023, 12, 31))

        tiles = list(source.load(anchor))  # lazy_load defaults False

        assert len(tiles) == 1
        assert len(calls) == 1
