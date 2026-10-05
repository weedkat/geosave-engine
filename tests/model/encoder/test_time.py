from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geosave_engine.model.encoder.time import time_labels


def _row(times):
    return pd.Series({"raster_metadata": {"image": {"times": times}}})


def test_time_labels_preserve_frame_order():
    actual = time_labels(_row(["2024-02-01", "2023-01-01"]))
    assert [value.isoformat() for value in actual] == [
        "2024-02-01T00:00:00",
        "2023-01-01T00:00:00",
    ]


@pytest.mark.parametrize("labels", [[], ["NaT"], [42], ["invalid"], [["2024-01-01"]]])
def test_time_labels_reject_invalid_coordinates(labels):
    with pytest.raises(ValueError, match="time"):
        time_labels(_row(labels))


def test_time_labels_accept_parquet_array_values():
    assert (
        time_labels(_row(np.array(["2024-01-02"])))[0].isoformat()
        == "2024-01-02T00:00:00"
    )
