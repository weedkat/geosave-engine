from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave_engine.ml.encoding.time import time_labels


def test_time_labels_preserve_frame_order() -> None:
    data = xr.DataArray(
        np.zeros((2, 1, 1)),
        dims=("time", "y", "x"),
        coords={
            "time": np.array(["2024-02-01", "2023-01-01"], dtype="datetime64[D]")
        },
    )

    actual = time_labels(data)

    assert [value.isoformat() for value in actual] == [
        "2024-02-01T00:00:00",
        "2023-01-01T00:00:00",
    ]


def test_time_labels_accept_a_scalar_coordinate() -> None:
    data = xr.DataArray(0, coords={"time": np.datetime64("2024-01-02")})

    assert time_labels(data)[0].isoformat() == "2024-01-02T00:00:00"


@pytest.mark.parametrize(
    "data",
    [
        xr.DataArray(0),
        xr.DataArray(np.empty((0,)), dims="time", coords={"time": []}),
        xr.DataArray(
            np.zeros((2, 2)),
            dims=("y", "x"),
            coords={
                "time": (
                    ("y", "x"),
                    np.full((2, 2), np.datetime64("2024-01-01")),
                )
            },
        ),
        xr.DataArray(0, coords={"time": 42}),
        xr.DataArray(0, coords={"time": np.datetime64("NaT", "ns")}),
    ],
)
def test_time_labels_reject_invalid_coordinates(data: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="time"):
        time_labels(data)


def test_time_labels_require_an_xarray_object() -> None:
    with pytest.raises(TypeError, match="Dataset or DataArray"):
        time_labels(object())  # type: ignore[arg-type]
