from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp

from geosave_engine.geodata.attrs import GDALVariable, GeoTIFFTags, read, rebase
from geosave_engine.geodata.utils.io import gdal
from geosave_engine.geodata.utils.io import geotiff

from .conftest import build_raster


def test_round_trip_keeps_the_variable_names(tmp_path: Path) -> None:
    written = build_raster(times=0)

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert list(restored.data_vars) == list(written.data_vars)
    assert restored.odc.geobox == written.odc.geobox
    for name, variable in written.data_vars.items():
        assert restored[name].dtype == np.dtype("uint16")
        assert np.array_equal(restored[name].values, variable.values)


def test_identity_and_the_cf_label_take_separate_slots(tmp_path: Path) -> None:
    written = build_raster(times=0)
    first = next(iter(written.data_vars))
    written[first].attrs["long_name"] = "Red reflectance (665 nm)"

    destination = geotiff.write_cog(written, tmp_path / "scene.tif")

    with rasterio.open(destination) as src:
        # The description is CF's label; identity rides in the band's own metadata.
        assert src.descriptions[0] == "Red reflectance (665 nm)"
        assert src.tags(1)["variable_name"] == first


def test_a_cf_long_name_survives_the_round_trip(tmp_path: Path) -> None:
    written = build_raster(times=0)
    first = next(iter(written.data_vars))
    written[first].attrs["long_name"] = "Red reflectance (665 nm)"

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored[first].attrs["long_name"] == "Red reflectance (665 nm)"


def test_a_scalar_time_survives_as_a_datetime_tag(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    written = build_raster(times=0).assign_coords(time=instant)

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.time.values == instant
    assert restored.attrs["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"


def test_a_time_dimension_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one GeoTIFF holds one grid"):
        geotiff.write_cog(build_raster(times=2), tmp_path / "scene.tif")


def test_sub_second_precision_refuses_rather_than_rounding(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08.500")
    written = build_raster(times=0).assign_coords(time=instant)

    with pytest.raises(ValueError, match="sub-second precision"):
        geotiff.write_cog(written, tmp_path / "scene.tif")


def test_the_time_coordinate_overrides_a_carried_datetime_tag(tmp_path: Path) -> None:
    written = rebase(
        build_raster(times=0).assign_coords(time=np.datetime64("2024-03-05T06:07:08")),
        GeoTIFFTags(TIFFTAG_DATETIME="1999:01:01 00:00:00"),
    )

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.attrs["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"


def test_carried_tags_survive_the_round_trip(tmp_path: Path) -> None:
    written = build_raster(times=0)
    written.attrs["TIFFTAG_ARTIST"] = "geosave"

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.attrs["TIFFTAG_ARTIST"] == "geosave"


def test_gtiff_takes_its_own_creation_options(tmp_path: Path) -> None:
    written = build_raster(times=0)

    destination = geotiff.write_gtiff(
        written, tmp_path / "scene.tif", tiled=True, blockxsize=128, blockysize=128
    )

    with rasterio.open(destination) as opened:
        assert opened.block_shapes[0] == (128, 128)


def test_a_chunked_cube_writes_window_by_window(tmp_path: Path) -> None:
    written = build_raster(times=0).chunk({"y": 2, "x": 2})

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    # Streaming is the point; the pixels have to survive it unchanged.
    assert list(restored.data_vars) == list(written.data_vars)
    for name, variable in written.data_vars.items():
        assert np.array_equal(restored[name].values, variable.values)


def test_destination_must_name_a_tiff(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in"):
        geotiff.write_cog(build_raster(times=0), tmp_path / "scene.nc")


def test_overwrite_guards_an_existing_file(tmp_path: Path) -> None:
    written = build_raster(times=0)
    destination = geotiff.write_cog(written, tmp_path / "scene.tif")

    with pytest.raises(FileExistsError, match="pass overwrite=True"):
        geotiff.write_cog(written, destination)

    assert geotiff.write_cog(written, destination, overwrite=True) == destination


def test_an_unnamed_band_keeps_rasterios_own_name(tmp_path: Path) -> None:
    plain = tmp_path / "plain.tif"
    with rasterio.open(
        plain, "w", driver="GTiff", height=2, width=2, count=2, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2, 2), "uint8"))
        destination.set_band_description(1, "Red reflectance")

    restored = gdal.read(plain)

    # A description is a label, never an identity, so neither band is renamed.
    assert list(restored.data_vars) == ["band_1", "band_2"]
    assert restored["band_1"].attrs["long_name"] == "Red reflectance"


def test_two_bands_naming_one_variable_refuse(tmp_path: Path) -> None:
    clashing = tmp_path / "clashing.tif"
    with rasterio.open(
        clashing, "w", driver="GTiff", height=2, width=2, count=2, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2, 2), "uint8"))
        for band in (1, 2):
            destination.update_tags(band, variable_name="red")

    with pytest.raises(ValueError, match="conflicts"):
        gdal.read(clashing)


@pytest.mark.parametrize("write", [geotiff.write_cog, geotiff.write_gtiff])
def test_colour_interpretation_round_trips(tmp_path: Path, write) -> None:
    written = build_raster(times=0)
    # uint16 with a role no TIFF photometric holds: the case a creation option
    # silently drops, so GDAL has to carry it in its own metadata instead.
    written = rebase(written, GDALVariable(colorinterp="red"), target="red")
    written = rebase(written, GDALVariable(colorinterp="nir"), target="nir")

    destination = write(written, tmp_path / "scene.tif")

    with rasterio.open(destination) as src:
        assert [band.name for band in src.colorinterp] == ["red", "nir"]
    header = read(gdal.read(destination))
    assert header.data_vars["red"].get(GDALVariable).colorinterp == "red"
    assert header.data_vars["nir"].get(GDALVariable).colorinterp == "nir"


def test_a_cube_measuring_nothing_leaves_gdal_to_interpret_its_bands(
    tmp_path: Path,
) -> None:
    destination = geotiff.write_cog(build_raster(times=0), tmp_path / "scene.tif")

    with rasterio.open(destination) as src:
        assert [band.name for band in src.colorinterp] == ["gray", "undefined"]


def test_colour_interpretation_leaves_a_cog_readable(tmp_path: Path) -> None:
    written = rebase(
        build_raster(times=0), GDALVariable(colorinterp="red"), target="red"
    )

    destination = geotiff.write_cog(written, tmp_path / "scene.tif")

    # A COG keeps its header at the front; reopening one to interpret its bands
    # would move the header to the end and cost the layout.
    with open(destination, "rb") as opened:
        header = opened.read(8)
    endian = "<" if header[:2] == b"II" else ">"
    (first_ifd,) = struct.unpack(f"{endian}I", header[4:8])
    assert first_ifd < destination.stat().st_size / 2
    with rasterio.open(destination) as src:
        assert src.block_shapes[0] != (src.height, src.width)


def test_colour_interpretation_lands_on_each_variable(tmp_path: Path) -> None:
    composite = tmp_path / "rgb.tif"
    with rasterio.open(
        composite, "w", driver="GTiff", height=2, width=2, count=3, dtype="uint8"
    ) as destination:
        destination.write(np.ones((3, 2, 2), "uint8"))
        destination.colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
        ]

    header = read(gdal.read(composite))

    assert [
        header.data_vars[name].get(GDALVariable).colorinterp
        for name in sorted(header.data_vars)
    ] == ["red", "green", "blue"]


def test_an_unreferenced_file_reads_unreferenced(tmp_path: Path) -> None:
    plain = tmp_path / "plain.tif"
    with rasterio.open(
        plain, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2), "uint8"), 1)

    assert gdal.read(plain).rio.crs is None
