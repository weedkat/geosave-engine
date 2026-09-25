from __future__ import annotations

import subprocess
import sys
from datetime import datetime as dt, timezone

import pytest
import pystac
import xarray as xr

from geosave_engine.geodata.attrs import CFVariable, Nodata
from geosave_engine.geodata.attrs.headers.stac import create_header


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
            media_type=pystac.MediaType.GEOTIFF,
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


def test_importing_attrs_does_not_import_stac_adapters() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import geosave_engine.geodata.attrs; "
            "assert 'geosave_engine.geodata.attrs.headers.stac' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_reading_uses_band_description_not_common_name() -> None:
    timestamp = dt(2025, 1, 1)
    item = _item(
        "scene-1",
        timestamp,
        {"common_name": "red", "description": "Surface reflectance", "unit": "1"},
    )

    header = create_header(
        [item], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    cf = header.data_vars["red"].get(CFVariable)
    assert cf == CFVariable(long_name="Surface reflectance", units="1")


def test_reading_skips_field_missing_from_one_item() -> None:
    first = dt(2025, 1, 1)
    second = dt(2025, 1, 2)
    items = [_item("scene-1", first, {"scale": 0.0001}), _item("scene-2", second, {})]

    header = create_header(
        items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(Nodata) is None


def test_reading_rejects_mixed_packing_values() -> None:
    first = dt(2025, 1, 1)
    second = dt(2025, 1, 2)
    items = [
        _item("scene-1", first, {"scale": 0.0001}),
        _item("scene-2", second, {"scale": 0.0002}),
    ]

    with pytest.raises(ValueError, match="'scale'"):
        create_header(
            items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
        )


def test_reading_rejects_an_empty_item_sequence() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        create_header([], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id")


def test_loader_nodata_is_authoritative_over_source_values() -> None:
    items = [
        _item("first", dt(2025, 1, 1), {"nodata": 0}),
        _item("second", dt(2025, 1, 2), {"nodata": 1}),
    ]
    loaded = xr.Dataset({"red": xr.DataArray([1.0], attrs={"nodata": -9999})})

    header = create_header(items, _collection(), loaded, groupby="id")

    assert header.data_vars["red"].get(Nodata).fill_value == -9999
