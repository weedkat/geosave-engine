"""Unit tests for datetime precision and filename range parsing."""

from datetime import datetime

import pytest

from geosave_engine.utils.datetime import date_range_from_path, parse_datetime_range


@pytest.mark.parametrize(
    ("value", "start", "end"),
    [
        ("2019", datetime(2019, 1, 1), datetime(2019, 12, 31, 23, 59, 59, 999999)),
        ("2019-05", datetime(2019, 5, 1), datetime(2019, 5, 31, 23, 59, 59, 999999)),
        ("2019-05-07", datetime(2019, 5, 7), datetime(2019, 5, 7, 23, 59, 59, 999999)),
        (
            "2019-05-07T10",
            datetime(2019, 5, 7, 10),
            datetime(2019, 5, 7, 10, 59, 59, 999999),
        ),
        (
            "2019-05-07T10:30",
            datetime(2019, 5, 7, 10, 30),
            datetime(2019, 5, 7, 10, 30, 59, 999999),
        ),
        (
            "2019-05-07T10:30:15",
            datetime(2019, 5, 7, 10, 30, 15),
            datetime(2019, 5, 7, 10, 30, 15, 999999),
        ),
    ],
)
def test_reduced_precision_expands_to_full_period(value, start, end):
    assert parse_datetime_range(value) == (start, end)


def test_datetime_object_remains_exact():
    value = datetime(2019, 5, 7)
    assert parse_datetime_range(value) == (value, value)


def test_explicit_reduced_precision_range_expands_both_bounds():
    assert parse_datetime_range(("2019-05", "2019-06")) == (
        datetime(2019, 5, 1),
        datetime(2019, 6, 30, 23, 59, 59, 999999),
    )


def test_date_range_from_path_reads_one_date_as_full_day():
    assert date_range_from_path("tile-20190507.tif") == (
        datetime(2019, 5, 7),
        datetime(2019, 5, 7, 23, 59, 59, 999999),
    )


def test_date_range_from_path_reads_start_and_end():
    assert date_range_from_path("tile-20190507-20190509.tif") == (
        datetime(2019, 5, 7),
        datetime(2019, 5, 9, 23, 59, 59, 999999),
    )


def test_date_range_from_path_allows_trailing_suffix_after_date():
    assert date_range_from_path("dw_107.9459135480_22.1422143633-20190923_consensus.tif") == (
        datetime(2019, 9, 23),
        datetime(2019, 9, 23, 23, 59, 59, 999999),
    )


def test_date_range_from_path_allows_trailing_suffix_after_range():
    assert date_range_from_path("tile-20190507-20190509_consensus.tif") == (
        datetime(2019, 5, 7),
        datetime(2019, 5, 9, 23, 59, 59, 999999),
    )


def test_date_range_from_path_rejects_nonstandard_name():
    with pytest.raises(ValueError, match="GeoTIFF filename must end"):
        date_range_from_path("tile_20190507.tif")


def test_reversed_range_raises():
    with pytest.raises(ValueError, match="before start"):
        parse_datetime_range(("2019-05-09", "2019-05-07"))
