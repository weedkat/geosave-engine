"""The attrs field helpers work for both merge and requirement consumers."""

from datetime import datetime

import numpy as np
from pydantic import ValidationError, field_validator
import pytest

from geosave_engine.geodata.attrs import AttrsModel, GDALVariable, GeoTIFFTags, Legend
from geosave_engine.geodata.attrs.model import (
    attrs_equal,
    common_attrs,
    parse_collection_text,
    parse_field_value,
)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (np.array([1, 2]), [1, 2], True),
        ((1, 2), np.array([1, 3]), False),
        (np.array([[1, 2]]), [1, 2], False),
        ({"values": np.array([1, np.nan])}, {"values": (1, float("nan"))}, True),
        ({"a": 1}, {"b": 1}, False),
        ([1, 2], [1], False),
        ("12", ["1", "2"], False),
        (np.int64(2), 2, True),
        (None, None, True),
        (float("nan"), float("nan"), True),
        (np.datetime64("2026-09-23", "ns"), np.datetime64("2026-09-23", "us"), True),
        (np.timedelta64(1, "s"), np.timedelta64(1_000_000_000, "ns"), True),
        (np.datetime64("NaT"), None, False),
        (np.empty((0, 2)), np.empty((0, 3)), False),
        (np.array(2), 2, True),
        (
            np.array(["2026-09-23"], dtype="datetime64[ns]"),
            np.array(["2026-09-23"], dtype="datetime64[us]"),
            True,
        ),
    ],
)
def test_native_metadata_values_compare_without_broadcasting(left, right, expected):
    assert attrs_equal(left, right) is expected
    assert attrs_equal(right, left) is expected


def test_nested_attrs_compare_inside_sequences():
    left = [{"values": np.array([1, np.nan])}]
    right = ({"values": [1, float("nan")]},)

    assert attrs_equal(left, right)
    assert attrs_equal(right, left)


def test_parse_field_value_keeps_annotated_constraints():
    with pytest.raises(ValidationError):
        parse_field_value(GDALVariable, "variable_name", "")


def test_parse_field_value_supports_partial_requirements_and_typed_timestamps():
    assert parse_field_value(Legend, "flag_values", "[0, 1]") == [0, 1]
    assert parse_field_value(
        GeoTIFFTags, "TIFFTAG_DATETIME", "2026-09-23T12:00:00"
    ) == datetime(2026, 9, 23, 12)


def test_parse_field_value_does_not_construct_a_partial_model():
    class DecoratedField(AttrsModel):
        NAME = "test_decorated_field"

        decorated_value: int

        @field_validator("decorated_value", mode="before")
        @classmethod
        def replace_value(cls, value):
            return 99

    assert parse_field_value(DecoratedField, "decorated_value", "4") == 4
    assert DecoratedField(decorated_value="4").decorated_value == 99


def test_parse_collection_text_only_decodes_text():
    native = {"values": [0, 1]}

    assert parse_collection_text(native) is native
    assert parse_collection_text('{"values":[0,1]}') == native
    assert parse_collection_text("not-json") == "not-json"


def test_invalid_collection_text_is_rejected_by_the_target_field():
    with pytest.raises(ValidationError):
        parse_field_value(Legend, "flag_values", "not-json")


def test_namespace_merge_preserves_equivalent_native_timestamps():
    from geosave_engine.geodata.attrs import AttrsNamespace

    timestamp = np.datetime64("2026-09-23", "ns")
    merged, dropped = AttrsNamespace.merge(
        [
            AttrsNamespace(foreign={"observed_at": timestamp}),
            AttrsNamespace(foreign={"observed_at": timestamp.astype("datetime64[us]")}
            ),
        ]
    )
    assert dropped == set()
    assert merged.foreign["observed_at"] == timestamp


def test_common_attrs_keeps_what_every_mapping_carries_alike():
    shared = common_attrs(
        [{"units": "1", "nodata": np.nan, "a": 1}, {"units": "1", "nodata": float("nan")}]
    )

    assert shared.keys() == {"units", "nodata"}
    assert np.isnan(shared["nodata"])
    assert common_attrs([{"units": "1"}, {"units": "K"}]) == {}
