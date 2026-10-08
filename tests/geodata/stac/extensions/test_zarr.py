"""The Zarr module states a store's layout through GeoSave's own class."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.stac.extensions import ZarrExtension, zarr


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_store_states_its_layout(scene, tmp_path) -> None:
    path = scene.gs.to_zarr(tmp_path / "cube.zarr")
    asset = _asset(str(path))

    with read_raster(path) as saved:
        zarr.write(asset, saved)

    store = ZarrExtension.ext(asset)
    assert (store.zarr_format, store.node_type, store.consolidated) == (
        3,
        "group",
        False,
    )
    assert asset.owner.stac_extensions == [ZarrExtension.get_schema_uri()]


def test_a_raster_not_opened_from_a_store_writes_nothing(scene) -> None:
    asset = _asset()

    zarr.write(asset, scene)

    assert asset.extra_fields == {}
    assert asset.owner.stac_extensions == []


def test_fields_read_back_through_the_class() -> None:
    asset = _asset("cube.zarr")
    ZarrExtension.ext(asset, add_if_missing=True).apply(
        zarr_format=2, node_type="group", consolidated=True
    )

    restored = pystac.Item.from_dict(asset.owner.to_dict()).assets["image"]

    assert asset.extra_fields == {
        "zarr:zarr_format": 2,
        "zarr:node_type": "group",
        "zarr:consolidated": True,
    }
    assert ZarrExtension.ext(restored).consolidated is True


def test_an_undeclared_extension_refuses() -> None:
    with pytest.raises(pystac.ExtensionNotImplemented):
        ZarrExtension.ext(_asset())
