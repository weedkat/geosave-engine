from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave_engine.geodata.attrs import BandStatistics


def test_gdals_own_strings_parse_into_numbers() -> None:
    summary = BandStatistics.model_validate(
        {
            "STATISTICS_MINIMUM": "1",
            "STATISTICS_MEAN": "2001.4357847178",
            "STATISTICS_APPROXIMATE": "YES",
        }
    )

    assert summary.STATISTICS_MINIMUM == 1.0
    assert summary.STATISTICS_MEAN == 2001.4357847178
    assert summary.STATISTICS_APPROXIMATE is True


def test_compute_summarises_only_the_present_pixels() -> None:
    array = xr.DataArray(np.array([[1.0, 2.0], [3.0, np.nan]]), dims=("y", "x"))

    summary = BandStatistics.compute(array)

    assert summary.STATISTICS_MINIMUM == 1.0
    assert summary.STATISTICS_MAXIMUM == 3.0
    assert summary.STATISTICS_MEAN == 2.0
    assert summary.STATISTICS_VALID_PERCENT == 75.0
    assert summary.STATISTICS_APPROXIMATE is False


def test_compute_refuses_a_variable_holding_no_present_pixel() -> None:
    absent = xr.DataArray(np.full((2, 2), np.nan), dims=("y", "x"), name="B04")

    with pytest.raises(ValueError, match="holds no present pixel"):
        BandStatistics.compute(absent)


def test_merging_keeps_the_extremes_of_the_joined_pixels() -> None:
    first = BandStatistics(STATISTICS_MINIMUM=1.0, STATISTICS_MAXIMUM=1000.0)
    second = BandStatistics(STATISTICS_MINIMUM=500.0, STATISTICS_MAXIMUM=1499.0)

    joined, dropped = BandStatistics.merge([first, second])

    assert joined.STATISTICS_MINIMUM == 1.0
    assert joined.STATISTICS_MAXIMUM == 1499.0
    assert not dropped


def test_a_minimum_of_zero_survives_the_join() -> None:
    # Zero is falsy, so filtering on truthiness would drop a real minimum.
    first = BandStatistics(STATISTICS_MINIMUM=0.0, STATISTICS_MAXIMUM=10.0)
    second = BandStatistics(STATISTICS_MINIMUM=3.0, STATISTICS_MAXIMUM=20.0)

    joined, _ = BandStatistics.merge([first, second])

    assert joined.STATISTICS_MINIMUM == 0.0


def test_merging_drops_what_a_join_cannot_recompute() -> None:
    first = BandStatistics(STATISTICS_MEAN=498.7, STATISTICS_VALID_PERCENT=100.0)
    second = BandStatistics(STATISTICS_MEAN=1001.6, STATISTICS_VALID_PERCENT=100.0)

    joined, dropped = BandStatistics.merge([first, second])

    # A mean needs each side's pixel count, which GDAL never wrote, and a share
    # that happens to agree still describes neither join.
    assert joined.STATISTICS_MEAN is None
    assert joined.STATISTICS_VALID_PERCENT is None
    assert dropped == {"STATISTICS_MEAN", "STATISTICS_VALID_PERCENT"}


def test_an_object_summarising_nothing_leaves_the_extremes_unknown() -> None:
    stated = BandStatistics(STATISTICS_MINIMUM=1.0, STATISTICS_MAXIMUM=10.0)

    joined, _ = BandStatistics.merge([stated, None])

    assert joined.STATISTICS_MINIMUM is None
    assert joined.STATISTICS_MAXIMUM is None


def test_merging_nothing_refuses() -> None:
    with pytest.raises(ValueError, match="needs at least one object"):
        BandStatistics.merge([])
