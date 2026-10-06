"""Filename tokens for instants."""

import pandas as pd

from geosave_engine.geodata.utils.datetime import format_instant


def test_format_instant_keeps_a_sub_second_fraction() -> None:
    assert format_instant(pd.Timestamp("2025-06-01T10:30:31")) == "20250601T103031"
    assert (
        format_instant(pd.Timestamp("2025-06-01T10:30:31.123456789"))
        == "20250601T103031_123456789"
    )
