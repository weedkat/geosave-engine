"""An Asset states what one saved raster file holds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import io, raster, read_raster
from geosave_engine.geodata.stac.asset import default_key, from_path, from_raster

from tests.geodata.conftest import build_raster


def test_a_cog_asset_states_what_the_file_holds(tmp_path: Path) -> None:
    source = build_raster(times=1, packed=True).isel(time=0)
    source["red"].attrs.update(units="1", long_name="Red reflectance")
    path = io.geotiff.write_cog(source, tmp_path / "scene.tif")

    fields = from_path(path).to_dict()

    assert fields["href"] == str(path)
    assert fields["type"] == pystac.MediaType.COG
    assert fields["roles"] == ["data"]
    assert fields["proj:code"] == "EPSG:32749"
    assert fields["proj:shape"] == [2, 2]
    assert fields["proj:transform"] == list(source.gs.geobox.transform)[:6]
    assert [band["name"] for band in fields["eo:bands"]] == ["red", "nir"]
    assert fields["eo:bands"][0]["description"] == "Red reflectance"
    red = fields["raster:bands"][0]
    assert (red["data_type"], red["nodata"], red["unit"]) == ("uint16", 0, "1")
    assert red["scale"] == pytest.approx(1e-4)
    assert fields["start_datetime"] == fields["end_datetime"] == "2025-06-01T00:00:00Z"
    orjson.dumps(fields)


def test_the_asset_describes_stored_values_not_decoded_ones(tmp_path: Path) -> None:
    first = io.geotiff.write_cog(
        build_raster(times=1, packed=True).isel(time=0), tmp_path / "packed.tif"
    )
    with read_raster(first, mask_and_scale=True) as decoded:
        assert decoded.red.dtype.kind == "f"
        second = io.geotiff.write_cog(decoded, tmp_path / "again.tif")

    red = from_path(second).to_dict()["raster:bands"][0]

    assert red["data_type"] == "uint16"
    assert red["scale"] == pytest.approx(1e-4)


def test_a_plain_geotiff_is_not_called_a_cog(tmp_path: Path) -> None:
    path = io.geotiff.write_gtiff(build_raster(), tmp_path / "plain.tif")

    assert from_path(path).media_type == pystac.MediaType.GEOTIFF


@pytest.mark.parametrize(
    ("write", "name", "media_type"),
    [
        (io.zarr.write, "cube.zarr", pystac.MediaType.ZARR),
        (io.netcdf.write, "cube.nc", pystac.MediaType.NETCDF),
    ],
)
def test_a_store_asset_spans_its_time_axis(tmp_path, write, name, media_type) -> None:
    source = build_raster(times=2)
    path = write(source, tmp_path / name)

    asset = from_path(path)

    start, end = source.gs.timespan
    assert asset.media_type == media_type
    assert asset.common_metadata.start_datetime.replace(tzinfo=None) == start
    assert asset.common_metadata.end_datetime.replace(tzinfo=None) == end
    assert [band["name"] for band in asset.to_dict()["eo:bands"]] == ["red", "nir"]


def test_a_timeless_file_states_no_time(tmp_path: Path) -> None:
    asset = from_path(io.geotiff.write_cog(build_raster(), tmp_path / "dem.tif"))

    assert asset.common_metadata.start_datetime is None


def test_a_relative_path_becomes_an_absolute_href(tmp_path, monkeypatch) -> None:
    io.geotiff.write_cog(build_raster(), tmp_path / "scene.tif")
    monkeypatch.chdir(tmp_path)

    assert from_path("scene.tif").href == str(tmp_path / "scene.tif")


def test_a_grid_without_an_epsg_code_states_its_wkt(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=12.34 +lon_0=56.78 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    path = io.geotiff.write_cog(
        raster({"class": np.ones((4, 4), dtype="uint8")}, grid), tmp_path / "laea.tif"
    )

    fields = from_path(path).to_dict()

    assert "proj:code" not in fields
    assert "Lambert" in fields["proj:wkt2"]


def test_a_raster_without_a_grid_refuses(tmp_path: Path) -> None:
    pixels = xr.Dataset({"image": (("y", "x"), np.arange(4).reshape(2, 2))})
    path = pixels.gs.to_netcdf(tmp_path / "pixels.nc")

    with pytest.raises(ValueError, match="no locatable grid"):
        from_path(path)


def test_a_folder_of_files_is_not_one_asset(tmp_path: Path) -> None:
    build_raster(times=2).gs.to_cog(tmp_path / "forest")

    with pytest.raises(ValueError, match="one raster file or store"):
        from_path(tmp_path / "forest")


# What decides how a reader opens the pixels; `description` and a zero `offset`
# are the two fields a COG reads back differently.
_PIXEL_FIELDS = (
    "href",
    "type",
    "roles",
    "proj:code",
    "proj:shape",
    "proj:transform",
    "start_datetime",
    "end_datetime",
)


def _band_facts(described: dict) -> list[tuple]:
    return [
        (raster.get("data_type"), raster.get("nodata"), raster.get("scale"), eo["name"])
        for raster, eo in zip(described["raster:bands"], described["eo:bands"])
    ]


@pytest.mark.parametrize("packed", [False, True])
def test_an_asset_built_at_write_time_agrees_with_the_file(tmp_path, packed) -> None:
    source = build_raster(times=2, packed=packed)
    scene = source.isel(time=0)[["red"]]
    (path, *_) = source.gs.to_cog(tmp_path / "forest", split_bands=True)
    store = source.gs.to_zarr(tmp_path / "forest.zarr")

    for built, read in [
        (from_raster(scene, path, driver="cog"), from_path(path)),
        (from_raster(source, store, driver="zarr"), from_path(store)),
    ]:
        built, read = built.to_dict(), read.to_dict()
        assert [built[key] for key in _PIXEL_FIELDS] == [
            read[key] for key in _PIXEL_FIELDS
        ]
        assert _band_facts(built) == _band_facts(read)


def test_from_raster_opens_no_file(tmp_path) -> None:
    missing = tmp_path / "never-written.zarr"

    described = from_raster(build_raster(times=2), missing, driver="zarr")

    assert described.href == str(missing)
    assert described.media_type == "application/vnd+zarr"


def test_a_whole_raster_is_keyed_by_its_variable_or_as_image() -> None:
    raster = build_raster()

    assert default_key(raster[["red"]]) == "red"
    assert default_key(raster) == "image"
