"""The Projection module writes a raster's grid."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import ProjectionExtension

from geosave_engine.geodata import raster
from geosave_engine.geodata.stac.extensions import projection


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_grid_is_written_through_the_projection_class(scene) -> None:
    asset = _asset()

    projection.write(asset, scene)

    grid = ProjectionExtension.ext(asset)
    assert (grid.code, grid.shape) == ("EPSG:32633", [64, 64])
    assert grid.transform == [10.0, 0.0, 300000.0, 0.0, -10.0, 5000640.0]
    assert asset.owner.stac_extensions == [ProjectionExtension.get_schema_uri()]


def test_a_grid_without_an_epsg_code_states_its_wkt() -> None:
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    grid = GeoBox.from_bbox((0, 0, 640, 640), crs, resolution=10)
    plain = raster({"height": (("y", "x"), np.zeros((64, 64), "float32"))}, grid)
    asset = _asset()

    projection.write(asset, plain)

    stated = ProjectionExtension.ext(asset)
    assert stated.code is None
    assert "Sinusoidal" in stated.wkt2


def test_a_raster_without_a_grid_refuses() -> None:
    bare = xr.Dataset({"height": (("y", "x"), np.zeros((2, 2), "float32"))})

    with pytest.raises(ValueError, match="no locatable grid"):
        projection.write(_asset(), bare)
