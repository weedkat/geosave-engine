"""Read ordered timestamps from a sample reference row."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def time_labels(row: pd.Series, *, raster: str = "image") -> list[datetime]:
    """Return this raster's ordered acquisition timestamps.

    Args:
        row: Sample row containing per-raster metadata.
        raster: Prepared raster whose frames the encoder receives.

    Returns:
        Datetimes in the same order as the model's temporal input.

    Raises:
        ValueError: Timestamps are missing, empty, or invalid.
    """
    labels = np.asarray(row["raster_metadata"][raster]["times"])
    if (
        labels.ndim != 1
        or not labels.size
        or not all(isinstance(label, str) for label in labels)
    ):
        raise ValueError("model context requires non-empty ISO time labels")
    try:
        values = labels.astype("datetime64[us]")
    except ValueError as error:
        raise ValueError("model context requires valid time labels") from error
    if np.isnat(values).any():
        raise ValueError("model context requires finite time labels")
    return [datetime.fromisoformat(str(value)) for value in values]
