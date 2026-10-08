"""The Classification module writes legends onto Raster bands."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest
from pystac.extensions.classification import ClassificationExtension
from pystac.extensions.raster import RasterExtension

from geosave_engine.geodata.attrs import Legend
from geosave_engine.geodata.stac.extensions import classification, raster

from tests.geodata.conftest import build_raster


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def _label(legend: Legend):
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    return label.gs.rebase(legend, target="label")


def test_a_legend_becomes_classes_on_its_band() -> None:
    label = _label(
        Legend(class_map={0: "background", 1: "forest"}, color_map={1: "#00ff00"})
    )
    asset = _asset()
    raster.write(asset, label)

    classification.write(asset, label)

    band = RasterExtension.ext(asset).bands[0]
    classes = ClassificationExtension.ext(band).classes
    assert [(entry.value, entry.name) for entry in classes] == [
        (0, "background"),
        (1, "forest"),
    ]
    assert classes[1].color_hint == "00FF00"
    assert ClassificationExtension.get_schema_uri() in asset.owner.stac_extensions


def test_a_raster_without_a_legend_declares_nothing(scene) -> None:
    asset = _asset()
    raster.write(asset, scene)

    classification.write(asset, scene)

    assert asset.owner.stac_extensions == [RasterExtension.get_schema_uri()]


def test_a_class_name_stac_refuses_raises() -> None:
    label = _label(Legend(class_map={1: "tree/shrub"}))
    asset = _asset()
    raster.write(asset, label)

    with pytest.raises(ValueError, match="tree/shrub"):
        classification.write(asset, label)


def test_bit_masks_write_no_classes() -> None:
    flags = build_raster()[["nir"]].astype("uint8").rename(nir="qa")
    flags["qa"].attrs.update(flag_masks=[1, 2], flag_meanings="cloud shadow")
    asset = _asset()
    raster.write(asset, flags)

    classification.write(asset, flags)

    assert RasterExtension.ext(asset).bands[0].to_dict() == {"data_type": "uint8"}


def test_classes_need_the_raster_bands_written_first() -> None:
    label = _label(Legend(class_map={1: "forest"}))
    asset = _asset()
    RasterExtension.add_to(asset.owner)

    with pytest.raises(ValueError):
        classification.write(asset, label)


def test_classes_are_read_back_as_the_legend_that_wrote_them() -> None:
    label = _label(
        Legend(class_map={0: "background", 1: "forest"}, color_map={1: "#00ff00"})
    )
    asset = _asset()
    raster.write(asset, label)
    classification.write(asset, label)

    (legend,) = classification.read(RasterExtension.ext(asset).bands[0])

    assert legend.class_map == {0: "background", 1: "forest"}
    assert legend.color_map[1] == "#00FF00"


def test_a_band_without_classes_reads_nothing(scene) -> None:
    asset = _asset()
    raster.write(asset, scene)

    assert classification.read(RasterExtension.ext(asset).bands[0]) == []
