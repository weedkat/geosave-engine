"""GeoHeader / GeoExtension: rebase semantics, hook dispatch, encode/decode, idempotency."""
from __future__ import annotations

import time as _time

import numpy as np
import pytest

from geosave_engine.geodata.extensions import Tags, TilingInfo, TimeSpan, TimeSpec
from geosave_engine.geodata.spatial.header import GeoHeader, decode_attrs, encode_attrs

from .conftest import make_anchor


class TestRebaseSemantics:
    def test_dict_merges_onto_current_fields(self):
        header = GeoHeader({"render": {"rgb_bands": ("B04", "B03", "B02")}})
        header = header.rebase(render={"class_map": {0: "bg"}})
        assert header.extensions["render"].rgb_bands == ("B04", "B03", "B02")
        assert header.extensions["render"].class_map == {0: "bg"}

    def test_instance_replaces_whole_namespace(self):
        header = GeoHeader({"tiling": TilingInfo(group_id="a", tile_id=0, data_shape=(4, 4), tile_shape=(2, 2), overlap=0, mode="edge")})
        replacement = TilingInfo(group_id="b", tile_id=1, data_shape=(8, 8), tile_shape=(4, 4), overlap=0, mode="edge")
        header = header.rebase(tiling=replacement)
        assert header.tiling is replacement

    def test_none_drops_the_namespace(self):
        header = GeoHeader({"tags": {"source": "survey"}})
        header = header.rebase(tags=None)
        assert header.tags == {}
        assert "tags" not in header.extensions

    def test_settable_false_raises(self):
        header = GeoHeader()
        with pytest.raises(ValueError, match="not settable through rebase"):
            header.rebase(timespec={"freq": "D"})

    def test_unregistered_namespace_raises(self):
        header = GeoHeader()
        with pytest.raises(Exception, match="No extension registered"):
            header.rebase(nonexistent={"x": 1})

    def test_no_kwargs_returns_self(self):
        header = GeoHeader({"tags": {"source": "survey"}})
        assert header.rebase() is header


class TestPositionalConstruction:
    def test_bare_dict_of_field_dicts(self):
        header = GeoHeader({"tags": {"source": "survey"}, "render": {"rgb_bands": ("B04", "B03", "B02")}})
        assert header.tags == {"source": "survey"}
        assert header.extensions["render"].rgb_bands == ("B04", "B03", "B02")

    def test_unregistered_namespace_dropped_with_warning(self):
        with pytest.warns(UserWarning, match="dropping unregistered extension namespace"):
            header = GeoHeader({"nonexistent": {"x": 1}})
        assert "nonexistent" not in header.extensions


class TestCombine:
    def test_timespec_keeps_when_all_equal(self):
        spec = TimeSpec(freq="D")
        merged = TimeSpec.combine([spec, spec, spec])
        assert merged == spec

    def test_timespec_drops_on_disagreement(self):
        merged = TimeSpec.combine([TimeSpec(freq="D"), TimeSpec(freq="ME")])
        assert merged is None

    def test_tiling_never_propagates(self):
        stamp = TilingInfo(group_id="a", tile_id=0, data_shape=(4, 4), tile_shape=(2, 2), overlap=0, mode="edge")
        assert TilingInfo.combine([stamp, stamp]) is None

    def test_timespan_never_propagates(self):
        span = TimeSpan.from_input("2024-01-15")
        assert TimeSpan.combine([span, span]) is None

    def test_default_rule_is_equal_or_raise(self):
        a = Tags(source="survey")
        b = Tags(source="different")
        assert Tags.combine([a, a]) == a
        with pytest.raises(ValueError, match="declares no merge rule"):
            Tags.combine([a, b])

    def test_geoheader_combine_dispatches_per_namespace(self):
        h1 = GeoHeader({"tags": {"source": "survey"}, "timespec": TimeSpec(freq="D")})
        h2 = GeoHeader({"tags": {"source": "survey"}, "timespec": TimeSpec(freq="D")})
        combined = GeoHeader.combine(h1, h2)
        assert combined.tags == {"source": "survey"}
        assert combined.timespec is not None and combined.timespec.freq == "D"


class TestEncodeDecodeRoundTrip:
    def test_json_encoding_keeps_nested_dicts(self):
        header = GeoHeader({"tags": {"source": "survey"}})
        encoded = encode_attrs({}, header, "json")
        assert encoded["tags"] == {"source": "survey"}
        assert isinstance(encoded["tags"], dict)

    def test_text_encoding_produces_one_json_string_per_namespace(self):
        header = GeoHeader({"tags": {"source": "survey"}})
        encoded = encode_attrs({}, header, "text")
        assert isinstance(encoded["tags"], str)

    def test_foreign_keys_survive_encode_and_decode(self):
        header = GeoHeader({"tags": {"source": "survey"}})
        encoded = encode_attrs({"mine": "keep"}, header, "json")
        assert encoded["mine"] == "keep"
        foreign, decoded = decode_attrs(encoded)
        assert foreign == {"mine": "keep"}
        assert decoded.tags == {"source": "survey"}

    def test_empty_namespace_is_omitted(self):
        header = GeoHeader({"tags": {}})
        encoded = encode_attrs({}, header, "json")
        assert "tags" not in encoded

    def test_round_trip_is_lossless_for_both_encodings(self):
        header = GeoHeader({
            "tags": {"source": "survey"},
            "render": {"rgb_bands": ("B04", "B03", "B02")},
        })
        for encoding in ("json", "text"):
            _, decoded = decode_attrs(encode_attrs({}, header, encoding))
            assert decoded.tags == header.tags
            assert decoded.extensions["render"].rgb_bands == header.extensions["render"].rgb_bands


class TestCheckHookAndEagerStamp:
    def test_render_missing_band_drops_with_warning(self):
        anchor = make_anchor(width=4, height=4).rebase(render={"rgb_bands": ("B04", "B08", "B02")})
        data = np.zeros((1, 4, 4), dtype="float32")
        with pytest.warns(UserWarning, match="render.rgb_bands references missing bands"):
            tile = anchor.to_geotile(data, bands=["B04"])
        assert tile.render is None

    def test_render_surviving_band_selection_keeps_hints(self):
        anchor = make_anchor(width=4, height=4).rebase(render={"rgb_bands": ("B04", "B03", "B02")})
        data = np.zeros((3, 4, 4), dtype="float32")
        tile = anchor.to_geotile(data, bands=["B04", "B03", "B02"])
        assert tile.render is not None
        assert tile.render.rgb_bands == ("B04", "B03", "B02")

    def test_data_attrs_mirrors_header_after_construction(self):
        anchor = make_anchor(width=4, height=4).rebase(tags={"source": "survey"})
        data = np.zeros((1, 4, 4), dtype="float32")
        tile = anchor.to_geotile(data, bands=["B04"])
        assert tile.data.attrs.get("tags") == {"source": "survey"}

    def test_stray_registered_key_in_raw_attrs_is_overwritten_not_merged(self):
        anchor = make_anchor(width=4, height=4).rebase(tags={"source": "survey"})
        data = np.zeros((1, 4, 4), dtype="float32")
        tile = anchor.to_geotile(data, bands=["B04"])
        stamped = tile.data.assign_attrs({"tags": {"stale": "leftover"}})
        rebuilt = tile.anchor.to_geotile(stamped)
        assert rebuilt.data.attrs["tags"] == {"source": "survey"}


class TestIdempotency:
    def test_repeated_check_on_valid_pair_is_a_noop(self):
        anchor = make_anchor(width=4, height=4).rebase(render={"rgb_bands": ("B04", "B03", "B02")})
        raw = np.zeros((3, 4, 4), dtype="float32")
        data = anchor.to_geotile(raw, bands=["B04", "B03", "B02"]).data
        first = GeoHeader(anchor.header.extensions, data=data)
        second = GeoHeader(first.extensions, data=data)
        assert first == second
        assert first.extensions["render"] == second.extensions["render"]

    def test_second_tile_construction_from_first_is_unchanged(self):
        anchor = make_anchor(width=4, height=4).rebase(tags={"source": "survey"})
        data = np.zeros((1, 4, 4), dtype="float32")
        tile = anchor.to_geotile(data, bands=["B04"])
        rebuilt = tile.anchor.to_geotile(tile.data)
        assert rebuilt.header == tile.header
        assert rebuilt.data.attrs == tile.data.attrs


class TestTimespanRename:
    def test_anchor_rebase_accepts_timespan_kwarg(self):
        anchor = make_anchor(time=None).rebase(timespan="2024-01-15")
        assert anchor.timespan is not None
        assert anchor.start.date().isoformat() == "2024-01-15"

    def test_old_time_kwarg_no_longer_recognized(self):
        anchor = make_anchor(time=None)
        with pytest.raises(Exception, match="No extension registered"):
            anchor.rebase(time="2024-01-15")

    def test_from_bbox_accepts_timespan_kwarg(self):
        from geosave_engine.geodata.spatial import GeoAnchor
        anchor = GeoAnchor.from_bbox((0, 0, 1, 1), resolution=0.5, crs="EPSG:4326", timespan="2024-01-15")
        assert anchor.timespan is not None


@pytest.mark.slow
class TestConstructionPerformance:
    def test_thousand_geotile_constructions_complete_quickly(self):
        """Guards against the eager-stamp-at-construction change adding real overhead."""
        anchor = make_anchor(width=8, height=8).rebase(
            tags={"source": "survey"},
            render={"rgb_bands": ("B04", "B03", "B02")},
        )
        data = np.zeros((3, 8, 8), dtype="float32")

        started = _time.perf_counter()
        for _ in range(1000):
            anchor.to_geotile(data, bands=["B04", "B03", "B02"])
        elapsed = _time.perf_counter() - started

        # generous bound — this is a regression guard, not a tight benchmark
        assert elapsed < 5.0, f"1000 GeoTile constructions took {elapsed:.2f}s"
