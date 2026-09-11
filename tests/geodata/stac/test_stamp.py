from __future__ import annotations

from datetime import datetime as dt, timezone

import pytest
import pystac
import xarray as xr

from geosave_engine.geodata.attrs import CFVariable, Packing, read
from geosave_engine.geodata.stac.stamp import stamp_stac


def _item(id: str, timestamp: dt, fields: dict[str, object]) -> pystac.Item:
    item = pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=timestamp.replace(tzinfo=timezone.utc),
        properties={},
    )
    item.add_asset(
        "red",
        pystac.Asset(
            href=f"https://example.com/{id}.tif",
            extra_fields={"raster:bands": [fields]},
        ),
    )
    return item


def _collection() -> pystac.Collection:
    return pystac.Collection(
        id="example",
        description="Example collection",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180, -90, 180, 90]]),
            pystac.TemporalExtent([[dt(2025, 1, 1, tzinfo=timezone.utc), None]]),
        ),
        license="CC-BY-4.0",
    )


def _loaded(*timestamps: dt) -> xr.Dataset:
    return xr.Dataset(
        {"red": ("time", range(len(timestamps)))},
        coords={"time": list(timestamps)},
    )


def test_stamp_uses_band_description_not_common_name() -> None:
    timestamp = dt(2025, 1, 1)
    item = _item(
        "scene-1",
        timestamp,
        {"common_name": "red", "description": "Surface reflectance", "unit": "1"},
    )

    stamped = stamp_stac(_loaded(timestamp), [item], _collection(), groupby="id")

    cf = read(stamped).data_vars["red"].get(CFVariable)
    assert cf == CFVariable(long_name="Surface reflectance", units="1")


def test_stamp_skips_field_missing_from_one_item() -> None:
    first = dt(2025, 1, 1)
    second = dt(2025, 1, 2)
    items = [_item("scene-1", first, {"scale": 0.0001}), _item("scene-2", second, {})]

    stamped = stamp_stac(_loaded(first, second), items, _collection(), groupby="id")

    assert read(stamped).data_vars["red"].get(Packing) is None


def test_stamp_rejects_mixed_nodata_values() -> None:
    first = dt(2025, 1, 1)
    second = dt(2025, 1, 2)
    items = [
        _item("scene-1", first, {"nodata": 0}),
        _item("scene-2", second, {"nodata": 1}),
    ]

    with pytest.raises(ValueError, match="'nodata'"):
        stamp_stac(_loaded(first, second), items, _collection(), groupby="id")


def test_stamp_rejects_an_empty_item_sequence() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        stamp_stac(_loaded(), [], _collection(), groupby="id")
