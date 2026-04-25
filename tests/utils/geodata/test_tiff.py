from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest

from geosave_engine.utils.geodata.tiff import parse_tiff_datetime, read_tiff_metadata


def test_parse_tiff_datetime_accepts_auxiliary_suffix() -> None:
    path = Path("._dw_107.9459135480_22.1422143633-20190923_consensus.tif")

    acquisition_dt = parse_tiff_datetime(path)

    assert acquisition_dt.year == 2019
    assert acquisition_dt.month == 9
    assert acquisition_dt.day == 23
    assert acquisition_dt.tzinfo == timezone.utc


def test_parse_tiff_datetime_accepts_canonical_filename() -> None:
    path = Path("dw_107.9459135480_22.1422143633-20190923.tif")

    acquisition_dt = parse_tiff_datetime(path)

    assert acquisition_dt.year == 2019
    assert acquisition_dt.month == 9
    assert acquisition_dt.day == 23
    assert acquisition_dt.tzinfo == timezone.utc


def test_read_tiff_metadata_rejects_appledouble_sidecar() -> None:
    path = Path("data/dynamicworld/val/__MACOSX/validation_set/expert_composites/EH/1/._dw_107.9459135480_22.1422143633-20190923_consensus.tif")

    with pytest.raises(ValueError, match="auxiliary macOS file is not a TIFF raster"):
        read_tiff_metadata(path)