from __future__ import annotations

from datetime import UTC, datetime

import pytest

from geosave_engine.geodata.attrs.models.stac import StacItem, StacMetadata
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.stac.item import ITEM_COLUMNS, item, sources

from tests.geodata.conftest import build_raster


def test_an_item_lists_its_columns_in_stac_order() -> None:
    row = item(build_raster(times=2), assets={"image": "image.tif"})

    assert list(row)[:6] == [
        "id",
        "type",
        "stac_version",
        "stac_extensions",
        "links",
        "datetime",
    ]
    assert list(row)[-2:] == ["assets", "sources"]
    assert set(row) <= set(ITEM_COLUMNS)
    assert row["proj:code"] == "EPSG:32749"
    assert tuple(row["proj:shape"]) == (2, 2)


def test_an_asset_is_described_from_the_raster_it_names() -> None:
    row = item(build_raster(times=1), assets={"image": "scene/image.tif"})

    assert row["assets"]["image"] == {
        "href": "scene/image.tif",
        "type": "image/tiff; application=geotiff",
        "roles": ["data"],
        "bands": [{"name": "red"}, {"name": "nir"}],
    }


def test_an_asset_key_names_a_layer_of_a_stack() -> None:
    raster = build_raster(times=1)
    sample = stack({"optical": raster[["red"]], "infrared": raster[["nir"]]})

    row = item(sample, assets={"optical": "optical.tif", "infrared": "infrared.tif"})

    assert row["assets"]["optical"]["bands"] == [{"name": "red"}]
    assert row["assets"]["infrared"]["bands"] == [{"name": "nir"}]
    with pytest.raises(ValueError, match="thermal"):
        item(sample, assets={"thermal": "thermal.tif"})


def test_an_asset_states_its_own_roles() -> None:
    row = item(
        build_raster(times=1),
        assets={"label": {"href": "label.tif", "roles": ["labels"]}},
    )

    assert row["assets"]["label"]["roles"] == ["labels"]
    assert row["assets"]["label"]["bands"] == [{"name": "red"}, {"name": "nir"}]
    with pytest.raises(ValueError, match="href"):
        item(build_raster(times=1), assets={"label": {"roles": ["labels"]}})


def test_a_timeless_item_needs_a_datetime() -> None:
    with pytest.raises(ValueError, match="datetime="):
        item(build_raster(), assets={"image": "image.tif"})

    row = item(
        build_raster(),
        assets={"image": "image.tif"},
        datetime=datetime(2025, 6, 1, tzinfo=UTC),
    )

    assert row["datetime"] == datetime(2025, 6, 1, tzinfo=UTC)
    assert row["start_datetime"] is None and row["end_datetime"] is None


def _loaded(prefix: str, days: tuple[int, ...], **properties: object) -> StacMetadata:
    return StacMetadata(
        stac_items=tuple(
            StacItem(
                id=f"{prefix}_{day:02d}",
                datetime=datetime(2025, 6, day, 2, 30),
                properties={"eo:cloud_cover": 10.0 * day, **properties},
            )
            for day in days
        )
    )


def test_sources_list_every_layers_provider_items_once() -> None:
    raster = build_raster(times=1)
    optical = raster[["red"]].gs.rebase(_loaded("S2A", (1, 3), platform="sentinel-2a"))
    radar = raster[["nir"]].gs.rebase(_loaded("S1A", (2,), platform="sentinel-1a"))

    found = sources(stack({"optical": optical, "radar": radar}))

    assert found == [
        {
            "eo:cloud_cover": 10.0,
            "platform": "sentinel-2a",
            "id": "S2A_01",
            "datetime": "2025-06-01T02:30:00",
        },
        {
            "eo:cloud_cover": 30.0,
            "platform": "sentinel-2a",
            "id": "S2A_03",
            "datetime": "2025-06-03T02:30:00",
        },
        {
            "eo:cloud_cover": 20.0,
            "platform": "sentinel-1a",
            "id": "S1A_02",
            "datetime": "2025-06-02T02:30:00",
        },
    ]


def test_a_raster_loaded_from_no_catalog_has_no_sources() -> None:
    assert sources(build_raster(times=1)) is None
