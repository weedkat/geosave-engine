from __future__ import annotations

import subprocess
import sys
from datetime import datetime as dt, timezone

import pytest
import pystac
import xarray as xr

from geosave_engine.geodata.attrs import CFVariable, Nodata, Packing
from geosave_engine.geodata.attrs.headers.stac import create_header
from geosave_engine.geodata.errors import DroppedAttrsWarning


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


@pytest.mark.parametrize("second", [{}, {"scale": 0.0002}])
def test_disagreeing_packing_drops_with_a_warning(second) -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"scale": 0.0001}),
        _item("scene-2", dt(2025, 1, 2), second),
    ]

    with pytest.warns(DroppedAttrsWarning, match="scale_factor") as warned:
        header = create_header(
            items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
        )

    assert header.data_vars["red"].get(Packing) is None
    assert "'scene-2'" in str(warned[0].message)


def test_an_item_without_the_asset_drops_its_fields_with_a_warning() -> None:
    first = _item("scene-1", dt(2025, 1, 1), {"unit": "1"})
    second = _item("scene-2", dt(2025, 1, 2), {"unit": "1"})
    del second.assets["red"]

    with pytest.warns(DroppedAttrsWarning, match="units"):
        header = create_header(
            [first, second], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
        )

    assert header.data_vars["red"].get(CFVariable) is None


def test_reading_writes_no_packing_when_no_item_publishes_it() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {"unit": "1"}),
    ]

    header = create_header(
        items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(Packing) is None


def test_reading_drops_a_label_missing_from_one_item() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {}),
    ]

    with pytest.warns(DroppedAttrsWarning, match="units"):
        header = create_header(
            items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
        )

    assert header.data_vars["red"].get(CFVariable) is None


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
