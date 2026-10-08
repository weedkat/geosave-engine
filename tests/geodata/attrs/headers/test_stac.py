from __future__ import annotations

import subprocess
import sys
from datetime import datetime as dt, timezone

import pytest
import pystac
import xarray as xr
from pystac.extensions.eo import Band
from pystac.extensions.raster import RasterBand

from geosave_engine.geodata.attrs import (
    CFVariable,
    Nodata,
    Packing,
    Spectral,
    StacMetadata,
)
from geosave_engine.geodata.attrs.headers.stac import (
    band_attrs,
    create_header,
    read_asset_fields,
    read_bands,
)
from geosave_engine.geodata.warnings import DroppedAttrsWarning


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


def test_importing_attrs_does_not_import_stac() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import geosave_engine.geodata.attrs; "
            "assert 'geosave_engine.geodata.stac' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_reading_uses_band_description_not_common_name() -> None:
    timestamp = dt(2025, 1, 1)
    item = _item("scene-1", timestamp, {"unit": "1"})
    item.assets["red"].extra_fields["eo:bands"] = [
        {"name": "B04", "common_name": "red", "description": "Surface reflectance"}
    ]

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
            [first, second],
            _collection(),
            xr.Dataset({"red": ("x", [1])}),
            groupby="id",
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


def test_reader_uses_the_selected_band_without_mutating_the_item() -> None:
    item = _item("multi", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "raster:bands": [{"unit": "1", "scale": 0.1}, {"unit": "K", "offset": 0}],
        "eo:bands": [{"description": "Reflectance"}, {"description": "Temperature"}],
    }
    before = item.to_dict()

    header = create_header(
        [item],
        _collection(),
        xr.Dataset({"red.1": ("x", [1]), "red.2": ("x", [2])}),
        groupby="id",
    )

    assert header.data_vars["red.1"].get(CFVariable) == CFVariable(
        units="1", long_name="Reflectance"
    )
    assert header.data_vars["red.1"].get(Packing) == Packing(scale_factor=0.1)
    assert header.data_vars["red.2"].get(CFVariable) == CFVariable(
        units="K", long_name="Temperature"
    )
    assert header.data_vars["red.2"].get(Packing) == Packing(add_offset=0)
    assert item.to_dict() == before


@pytest.mark.parametrize("selection", [None, (), ("unit", "missing")])
def test_captured_properties_select_only_requested_fields(selection) -> None:
    item = _item("scene", dt(2025, 1, 1), {"unit": "1", "scale": 0.1})
    item.properties["platform"] = "example"

    header = create_header(
        [item],
        _collection(),
        xr.Dataset({"red": ("x", [1])}),
        groupby="id",
        item_properties=("platform", "missing"),
        asset_fields=selection,
    )

    metadata = header.root.get(StacMetadata)
    assert metadata.stac_items[0].properties == {"platform": "example"}
    expected = (
        {"unit": "1", "scale": 0.1}
        if selection is None
        else ({} if not selection else {"unit": "1"})
    )
    assert metadata.stac_items[0].assets == ({"red": expected} if expected else {})


def test_asset_reader_keeps_core_band_precedence() -> None:
    asset = pystac.Asset(
        "https://example.com/a.tif",
        extra_fields={
            "unit": "asset",
            "description": "asset",
            "raster:bands": [{"unit": "raster"}],
            "eo:bands": [{"description": "EO"}],
            "bands": [{"unit": "core", "description": "Core"}],
        },
    )

    assert read_asset_fields(asset) == {"unit": "core", "description": "Core"}


def test_loader_packing_overrides_the_agreed_source_packing() -> None:
    item = _item("scene", dt(2025, 1, 1), {"scale": 0.1, "offset": 2})
    loaded = xr.Dataset({"red": xr.DataArray([1], attrs={"scale_factor": 1.0})})

    header = create_header([item], _collection(), loaded, groupby="id")

    assert header.data_vars["red"].get(Packing) == Packing(scale_factor=1, add_offset=2)


def test_spectral_facts_reach_the_variable() -> None:
    item = _item("scene", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "eo:bands": [{"name": "B04", "common_name": "red", "center_wavelength": 0.665}]
    }

    header = create_header(
        [item], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(Spectral) == Spectral(
        common_name="red", center_wavelength=0.665
    )


def test_core_bands_read_like_legacy_ones() -> None:
    item = _item("scene", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "bands": [{"name": "B04", "unit": "1", "raster:scale": 0.0001}]
    }

    header = create_header(
        [item], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(CFVariable) == CFVariable(units="1")
    assert header.data_vars["red"].get(Packing) == Packing(scale_factor=0.0001)


def test_a_band_states_units_and_packing_as_attrs() -> None:
    raster = RasterBand.create(nodata=0, unit="1", scale=0.0001)
    spectral = Band.create(name="red", description="Red")

    assert band_attrs(raster, spectral) == {
        "units": "1",
        "long_name": "Red",
        "scale_factor": 0.0001,
    }


def test_a_bare_band_states_no_attrs() -> None:
    assert band_attrs(RasterBand({"data_type": "float32"}), Band({"name": "dem"})) == {}


def test_legacy_listings_pair_by_index() -> None:
    asset = pystac.Asset(
        "a.tif",
        extra_fields={
            "raster:bands": [{"data_type": "uint16", "nodata": 0, "scale": 0.0001}],
            "eo:bands": [{"name": "B04", "common_name": "red"}, {"name": "B08"}],
        },
    )

    (first, red), (second, nir) = read_bands(asset)

    assert (first.scale, red.common_name) == (0.0001, "red")
    assert (second.to_dict(), nir.name) == ({}, "B08")


def test_core_bands_read_without_parquet_nulls() -> None:
    asset = pystac.Asset(
        "a.tif",
        extra_fields={
            "bands": [{"name": "label", "unit": None, "raster:scale": None}],
            "raster:bands": [{"unit": "ignored"}],
        },
    )

    ((raster, spectral),) = read_bands(asset)

    assert (raster.to_dict(), spectral.to_dict()) == ({}, {"name": "label"})


def test_an_asset_without_bands_reads_as_none() -> None:
    assert read_bands(pystac.Asset("thumbnail.png")) == []
