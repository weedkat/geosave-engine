"""Unit tests for datetime precision and filename range parsing."""

from datetime import datetime

import pytest

from geosave_engine.geodata.utils.datetime import date_range_from_path, date_range_to_suffix, parse_datetime_range


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


def test_date_range_from_path_reads_second_precision():
    assert date_range_from_path("tile-20190507T103015.tif") == (
        datetime(2019, 5, 7, 10, 30, 15),
        datetime(2019, 5, 7, 10, 30, 15, 999999),
    )


def test_date_range_from_path_reads_second_precision_range():
    assert date_range_from_path("tile-20190507T103015-20190507T104512.tif") == (
        datetime(2019, 5, 7, 10, 30, 15),
        datetime(2019, 5, 7, 10, 45, 12, 999999),
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
    with pytest.raises(ValueError, match="Filename must end"):
        date_range_from_path("tile_20190507.tif")


def test_reversed_range_raises():
    with pytest.raises(ValueError, match="before start"):
        parse_datetime_range(("2019-05-09", "2019-05-07"))


def test_compact_form_accepted_directly():
    assert parse_datetime_range("20190507") == parse_datetime_range("2019-05-07")
    assert parse_datetime_range("20190507T103015") == parse_datetime_range("2019-05-07T10:30:15")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019", "2019"),
        ("2019-05", "201905"),
        ("2019-05-07", "20190507"),
        ("2019-05-07T10", "20190507T10"),
        ("2019-05-07T10:30", "20190507T1030"),
        ("2019-05-07T10:30:15", "20190507T103015"),
    ],
)
def test_date_range_to_suffix_round_trips_single_token(value, expected):
    assert date_range_to_suffix(parse_datetime_range(value)) == expected


def test_date_range_to_suffix_arbitrary_range_uses_two_tokens():
    value = (datetime(2019, 5, 7), datetime(2019, 5, 9, 23, 59, 59, 999999))
    assert date_range_to_suffix(value) == "20190507-20190509"


def test_date_range_to_suffix_round_trips_through_date_range_from_path(tmp_path):
    original = date_range_from_path("tile-20190507T103015-20190507T104512.tif")
    suffix = date_range_to_suffix(original)
    assert date_range_from_path(f"tile-{suffix}.tif") == original
