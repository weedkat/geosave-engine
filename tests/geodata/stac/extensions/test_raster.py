"""The Raster module writes how each band is typed, filled and packed."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import dask
import numpy as np
import pytest
from pystac.extensions.raster import RasterBand, RasterExtension

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.attrs import CFVariable, Packing
from geosave_engine.geodata.stac.extensions import raster

from tests.geodata.conftest import build_raster


def _refuse(*args, **kwargs):
    raise AssertionError("writing the Raster fields computed pixels")


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_saved_cog_states_what_it_stores(scene, tmp_path) -> None:
    path = scene.gs.to_cog(tmp_path / "forest")[0]
    asset = _asset(str(path))

    with read_raster(path) as saved, dask.config.set(scheduler=_refuse):
        raster.write(asset, saved)

    bands = RasterExtension.ext(asset).bands
    assert [band.data_type for band in bands] == ["uint16", "uint16"]
    assert bands[0].nodata == 0
    assert asset.owner.stac_extensions == [RasterExtension.get_schema_uri()]


def test_packing_and_units_reach_the_band() -> None:
    scene = build_raster(packed=True).gs.rebase(CFVariable(units="1"), target="red")
    asset = _asset()

    raster.write(asset, scene)

    red = RasterExtension.ext(asset).bands[0]
    assert (red.scale, red.offset, red.unit) == (pytest.approx(1e-4), 0.0, "1")


def test_a_plain_raster_states_only_its_stored_type() -> None:
    dem = build_raster()[["nir"]].astype("float32").rename(nir="height")
    asset = _asset()

    raster.write(asset, dem)

    assert [band.to_dict() for band in RasterExtension.ext(asset).bands] == [
        {"data_type": "float32"}
    ]


def test_a_dtype_stac_cannot_name_is_other() -> None:
    mask = build_raster()[["nir"]].astype(bool).rename(nir="mask")
    asset = _asset()

    raster.write(asset, mask)

    assert RasterExtension.ext(asset).bands[0].data_type == "other"


@pytest.mark.parametrize(
    ("fill", "expected"), [(np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf")]
)
def test_a_nonfinite_fill_uses_stac_strings(fill, expected) -> None:
    dem = build_raster()[["nir"]].astype("float32").rename(nir="height")
    dem["height"].attrs["_FillValue"] = fill
    asset = _asset()

    raster.write(asset, dem)

    assert RasterExtension.ext(asset).bands[0].nodata == expected


def test_packing_and_units_are_read_from_a_band() -> None:
    band = RasterBand.create(nodata=0, unit="1", scale=0.0001, offset=-0.1)

    assert raster.read(band) == [
        Packing(scale_factor=0.0001, add_offset=-0.1),
        CFVariable(units="1"),
    ]


def test_a_band_stating_only_its_type_reads_nothing() -> None:
    assert raster.read(RasterBand({"data_type": "float32", "nodata": 0})) == []
