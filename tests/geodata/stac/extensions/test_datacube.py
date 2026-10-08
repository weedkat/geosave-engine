"""The Datacube module states a raster's time labels."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
from pystac.extensions.datacube import DatacubeExtension

from geosave_engine.geodata.stac.extensions import datacube


def _asset() -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset("cube.zarr", roles=["data"]))
    return item.assets["image"]


def test_a_cube_states_every_time_label(scene) -> None:
    asset = _asset()

    datacube.write(asset, scene)

    time = DatacubeExtension.ext(asset).dimensions["time"]
    assert time.values == ["2025-06-01T00:00:00Z", "2025-06-11T00:00:00Z"]
    assert time.extent == ["2025-06-01T00:00:00Z", "2025-06-11T00:00:00Z"]
    assert asset.owner.stac_extensions == [DatacubeExtension.get_schema_uri()]


def test_a_single_date_scene_states_its_one_label(scene) -> None:
    asset = _asset()

    datacube.write(asset, scene.isel(time=1))

    assert DatacubeExtension.ext(asset).dimensions["time"].values == [
        "2025-06-11T00:00:00Z"
    ]


def test_a_timeless_raster_writes_nothing(label) -> None:
    asset = _asset()

    datacube.write(asset, label)

    assert asset.extra_fields == {}
    assert asset.owner.stac_extensions == []
