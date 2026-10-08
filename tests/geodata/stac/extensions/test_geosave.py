"""GeoSave's own Item properties go through its extension class."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest

from geosave_engine.geodata.stac.extensions import GeosaveExtension


def _item() -> pystac.Item:
    return pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})


def test_the_stack_is_an_item_property() -> None:
    item = _item()

    GeosaveExtension.ext(item, add_if_missing=True).stack = "s0"

    assert item.properties == {"geosave:stack": "s0"}
    assert item.stac_extensions == [GeosaveExtension.get_schema_uri()]
    assert GeosaveExtension.ext(pystac.Item.from_dict(item.to_dict())).stack == "s0"


def test_an_undeclared_extension_refuses() -> None:
    with pytest.raises(pystac.ExtensionNotImplemented):
        GeosaveExtension.ext(_item())
