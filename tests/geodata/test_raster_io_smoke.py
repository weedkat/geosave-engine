from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from geosave_engine.geodata.attrs import GeoTIFFTags, rebase
from geosave_engine.geodata.utils.io import gdal
from geosave_engine.geodata.utils.io import geotiff

from .conftest import build_raster


def test_round_trip_keeps_bands_named_after_variables(tmp_path: Path) -> None:
    written = build_raster(times=0).to_dataarray(dim="band")

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert list(restored.band.values) == list(written.band.values)
    assert restored.odc.geobox == written.odc.geobox
    assert restored.dtype == np.dtype("uint16")
    assert np.array_equal(restored.values, written.values)


def test_a_scalar_time_survives_as_a_datetime_tag(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    written = build_raster(times=0).to_dataarray(dim="band").assign_coords(time=instant)

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.time.values == instant
    assert restored.attrs["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"


def test_a_time_dimension_refuses(tmp_path: Path) -> None:
    written = build_raster(times=2).to_dataarray(dim="band")

    with pytest.raises(ValueError, match="one GeoTIFF holds one grid"):
        geotiff.write_cog(written, tmp_path / "scene.tif")


def test_sub_second_precision_refuses_rather_than_rounding(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08.500")
    written = build_raster(times=0).to_dataarray(dim="band").assign_coords(time=instant)

    with pytest.raises(ValueError, match="sub-second precision"):
        geotiff.write_cog(written, tmp_path / "scene.tif")


def test_the_time_coordinate_overrides_a_carried_datetime_tag(tmp_path: Path) -> None:
    written = rebase(
        build_raster(times=0)
        .to_dataarray(dim="band")
        .assign_coords(time=np.datetime64("2024-03-05T06:07:08")),
        GeoTIFFTags(TIFFTAG_DATETIME="1999:01:01 00:00:00"),
    )

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.attrs["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"


def test_carried_tags_survive_the_round_trip(tmp_path: Path) -> None:
    written = build_raster(times=0).to_dataarray(dim="band")
    written.attrs["TIFFTAG_ARTIST"] = "geosave"

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.attrs["TIFFTAG_ARTIST"] == "geosave"


def test_gtiff_takes_its_own_creation_options(tmp_path: Path) -> None:
    written = build_raster(times=0).to_dataarray(dim="band")

    destination = geotiff.write_gtiff(
        written, tmp_path / "scene.tif", tiled=True, blockxsize=128, blockysize=128
    )

    with rasterio.open(destination) as opened:
        assert opened.block_shapes[0] == (128, 128)
        assert opened.descriptions == tuple(str(n) for n in written.band.values)


def test_destination_must_name_a_tiff(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in"):
        geotiff.write_cog(
            build_raster(times=0).to_dataarray(dim="band"), tmp_path / "scene.nc"
        )


def test_overwrite_guards_an_existing_file(tmp_path: Path) -> None:
    written = build_raster(times=0).to_dataarray(dim="band")
    destination = geotiff.write_cog(written, tmp_path / "scene.tif")

    with pytest.raises(FileExistsError, match="pass overwrite=True"):
        geotiff.write_cog(written, destination)

    assert geotiff.write_cog(written, destination, overwrite=True) == destination


def test_an_undescribed_band_keeps_its_index(tmp_path: Path) -> None:
    plain = tmp_path / "plain.tif"
    with rasterio.open(
        plain, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2), "uint8"), 1)

    restored = gdal.read(plain)

    assert list(restored.band.values) == [1]


def test_band_names_replace_the_file_names(tmp_path: Path) -> None:
    destination = geotiff.write_cog(
        build_raster(times=0).to_dataarray(dim="band"), tmp_path / "scene.tif"
    )

    assert list(gdal.read(destination, band_names=["a", "b"]).band.values) == [
        "a",
        "b",
    ]

    with pytest.raises(ValueError, match="name every band exactly once"):
        gdal.read(destination, band_names=["a"])


def test_an_unreferenced_file_reads_unreferenced(tmp_path: Path) -> None:
    plain = tmp_path / "plain.tif"
    with rasterio.open(
        plain, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2), "uint8"), 1)

    assert gdal.read(plain).rio.crs is None
