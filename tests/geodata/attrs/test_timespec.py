from __future__ import annotations

from datetime import datetime as dt

import numpy as np

from geosave_engine.geodata.attrs import TimeSpec


def test_bounds_are_half_open_forward_for_start_labels() -> None:
    spec = TimeSpec.from_resample("MS")
    labels = np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")

    edges = spec.bounds(labels)

    assert edges.dtype == np.dtype("datetime64[us]")
    assert edges.tolist() == [
        [dt(2024, 1, 1), dt(2024, 2, 1)],
        [dt(2024, 2, 1), dt(2024, 3, 1)],
    ]


def test_bounds_run_backward_for_end_labels() -> None:
    spec = TimeSpec.from_resample("ME")
    labels = np.array(["2024-01-31", "2024-02-29"], dtype="datetime64[ns]")

    edges = spec.bounds(labels)

    assert edges.tolist() == [
        [dt(2023, 12, 31), dt(2024, 1, 31)],
        [dt(2024, 1, 31), dt(2024, 2, 29)],
    ]


def test_bounds_place_each_label_independently_of_gaps() -> None:
    spec = TimeSpec.from_resample("MS")
    labels = np.array(["2024-01-01", "2024-04-01"], dtype="datetime64[ns]")

    edges = spec.bounds(labels)

    assert edges.tolist() == [
        [dt(2024, 1, 1), dt(2024, 2, 1)],
        [dt(2024, 4, 1), dt(2024, 5, 1)],
    ]


def test_bounds_step_by_the_full_multiple() -> None:
    spec = TimeSpec.from_resample("5D")
    labels = np.array(["2024-01-01", "2024-01-06"], dtype="datetime64[ns]")

    edges = spec.bounds(labels)

    assert edges.tolist() == [
        [dt(2024, 1, 1), dt(2024, 1, 6)],
        [dt(2024, 1, 6), dt(2024, 1, 11)],
    ]
