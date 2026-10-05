from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from geosave_engine.geodata.attrs import (
    ACDD,
    AttrsHeader,
    CFVariable,
    GDALVariable,
    GeoTIFFTags,
    Legend,
    Packing,
    Nodata,
    create_header,
    rebase,
)
from geosave_engine.geodata.errors import UnreadMaskWarning
from geosave_engine.geodata.io import gdal
from geosave_engine.geodata.io import geotiff

from tests.geodata.conftest import build_raster


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


@pytest.mark.parametrize(
    ("stem", "instant"),
    [
        ("20190507_red", "2019-05-07T00:00:00"),
        ("20250601T000000_red", "2025-06-01T00:00:00"),
        ("20190507-20190509_red", "2019-05-07T00:00:00"),
    ],
    ids=["a day", "an instant", "a period takes its first instant"],
)
def test_a_dateless_file_takes_the_time_its_name_spells(
    tmp_path: Path, stem: str, instant: str
) -> None:
    written = build_raster(times=0)

    restored = gdal.read(geotiff.write_cog(written, tmp_path / f"{stem}.tif"))

    assert restored.time.values == np.datetime64(instant)


def test_a_file_naming_no_date_carries_no_time(tmp_path: Path) -> None:
    written = build_raster(times=0)

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "dem.tif"))

    assert "time" not in restored.coords


def test_the_datetime_tag_outranks_the_name(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    written = build_raster(times=0).assign_coords(time=instant)

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "20190507_red.tif"))

    assert restored.time.values == instant


def test_several_instants_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one GeoTIFF holds one instant"):
        geotiff.write_cog(build_raster(times=2), tmp_path / "scene.tif")


def test_one_instant_writes_however_the_axis_spells_it(tmp_path: Path) -> None:
    along_an_axis = geotiff.write_cog(build_raster(times=1), tmp_path / "axis.tif")
    as_a_scalar = geotiff.write_cog(
        build_raster(times=1).isel(time=0), tmp_path / "scalar.tif"
    )

    assert gdal.read(along_an_axis).time == gdal.read(as_a_scalar).time


def test_sub_second_time_survives_in_geosave_metadata(tmp_path: Path) -> None:
    instant = np.datetime64("2018-12-26T10:34:39.024000001")
    written = build_raster(times=0).assign_coords(time=instant)

    destination = geotiff.write_cog(written, tmp_path / "20190507_red.tif")
    restored = gdal.read(destination)

    with rasterio.open(destination) as src:
        assert src.tags()["TIFFTAG_DATETIME"] == "2018:12:26 10:34:39"
        assert src.tags()["GEOSAVE_DATETIME"] == "2018-12-26T10:34:39.024000001"
    assert restored.time.values == instant


def test_geosave_datetime_needs_a_valid_instant() -> None:
    with pytest.raises(ValueError, match="GEOSAVE_DATETIME needs an ISO 8601 instant"):
        GeoTIFFTags(GEOSAVE_DATETIME="not-a-datetime")


def test_the_standard_datetime_tag_remains_the_fallback(tmp_path: Path) -> None:
    path = tmp_path / "20190507_red.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2), "uint8"), 1)
        destination.update_tags(TIFFTAG_DATETIME="2024:03:05 06:07:08")

    restored = gdal.read(path)

    assert restored.time.values == np.datetime64("2024-03-05T06:07:08")


def test_the_time_coordinate_overrides_a_carried_datetime_tag(tmp_path: Path) -> None:
    written = rebase(
        build_raster(times=0).assign_coords(time=np.datetime64("2024-03-05T06:07:08")),
        GeoTIFFTags(TIFFTAG_DATETIME="1999:01:01 00:00:00"),
    )

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.attrs["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"


def test_geotiff_tags_take_description_and_time_from_the_raster() -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    source = rebase(
        build_raster(times=0).assign_coords(time=instant),
        ACDD(summary="Surface reflectance scene"),
        GeoTIFFTags(
            TIFFTAG_IMAGEDESCRIPTION="stale",
            TIFFTAG_DATETIME="1999:01:01 00:00:00",
            TIFFTAG_ARTIST="GeoSave",
        ),
    )

    tags = GeoTIFFTags.from_xarray(source)

    assert tags.TIFFTAG_IMAGEDESCRIPTION == "Surface reflectance scene"
    assert tags.TIFFTAG_DATETIME == instant.astype("datetime64[s]").astype(object)
    assert tags.TIFFTAG_ARTIST == "GeoSave"


def test_geotiff_tags_keep_a_description_when_acdd_has_no_summary() -> None:
    source = rebase(
        build_raster(times=0),
        ACDD(title="A scene"),
        GeoTIFFTags(TIFFTAG_IMAGEDESCRIPTION="Hand-authored description"),
    )

    tags = GeoTIFFTags.from_xarray(source)

    assert tags.TIFFTAG_IMAGEDESCRIPTION == "Hand-authored description"


@pytest.mark.parametrize(
    ("crs", "expected"),
    [
        ("EPSG:32749", 10.0),
        ("EPSG:2277", 32.808333333333),
    ],
    ids=["metres", "us-survey-feet"],
)
def test_geotiff_tags_calculate_pixels_per_centimetre(crs: str, expected: float):
    tags = GeoTIFFTags.from_xarray(build_raster(crs=crs), map_scale=10_000)

    assert tags.TIFFTAG_XRESOLUTION == pytest.approx(expected)
    assert tags.TIFFTAG_YRESOLUTION == pytest.approx(expected)
    assert tags.TIFFTAG_RESOLUTIONUNIT == 3


@pytest.mark.parametrize("map_scale", [0, -1, float("nan"), float("inf")])
def test_geotiff_tags_refuse_an_invalid_map_scale(map_scale: float) -> None:
    with pytest.raises(ValueError, match="map_scale"):
        GeoTIFFTags.from_xarray(build_raster(), map_scale=map_scale)


def test_geotiff_tags_refuse_physical_resolution_for_a_geographic_grid() -> None:
    with pytest.raises(ValueError, match="projected CRS"):
        GeoTIFFTags.from_xarray(build_raster(crs="EPSG:4326"), map_scale=10_000)


def test_geotiff_tags_refuse_physical_resolution_without_a_grid() -> None:
    source = xr.Dataset({"red": (("y", "x"), np.ones((2, 2), dtype="uint8"))})

    with pytest.raises(ValueError, match="regular grid"):
        GeoTIFFTags.from_xarray(source, map_scale=10_000)


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

    with pytest.raises(ValueError, match="already names"):
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
    header = create_header(gdal.read(destination))
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

    header = create_header(gdal.read(composite))

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


def test_bands_declaring_different_absent_pixels_refuse_to_write(
    tmp_path: Path,
) -> None:
    written = rebase(build_raster(), Nodata(fill_value=0), target="red")

    # GDAL writes one fill value for the file, so rioxarray never sees the mix.
    with pytest.raises(ValueError, match="one GeoTIFF holds one fill value"):
        geotiff.write_cog(written, tmp_path / "scene.tif")


def test_bands_sharing_one_fill_value_write(tmp_path: Path) -> None:
    written = rebase(build_raster(), Nodata(fill_value=0), target=["red", "nir"])

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.red.attrs["_FillValue"] == 0
    assert restored.nir.attrs["_FillValue"] == 0


def test_a_units_string_reaches_gdals_own_band_unit(tmp_path: Path) -> None:
    written = rebase(build_raster(), CFVariable(units="1"), target=["red", "nir"])

    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    # A tag is text a reader must look for; gdalinfo reads the band's unit.
    with rasterio.open(path) as src:
        assert src.units == ("1", "1")


def test_stacked_array_restores_each_bands_metadata_for_geotiff(
    tmp_path: Path,
) -> None:
    written = build_raster(times=0)
    written = rebase(
        written,
        CFVariable(units="reflectance"),
        Packing(scale_factor=1e-4),
        GDALVariable(colorinterp="red"),
        target="red",
    )
    written = rebase(
        written,
        CFVariable(units="index"),
        Packing(scale_factor=2e-4),
        GDALVariable(colorinterp="nir"),
        target="nir",
    )

    path = written.gs.to_array().gs.to_cog(tmp_path / "stacked.tif")

    with rasterio.open(path) as src:
        assert src.units == ("reflectance", "index")
        assert src.scales == pytest.approx((1e-4, 2e-4))
        assert src.colorinterp == (ColorInterp.red, ColorInterp.nir)


def test_a_legend_colour_map_writes_a_gdal_palette(tmp_path: Path) -> None:
    labelled = build_raster()[["red"]].astype("uint8")
    written = rebase(
        labelled,
        Legend(class_map={0: "bg", 1: "palm"}),
        Legend(color_map={0: "#000000", 1: "#00ff00"}),
        target="red",
    )

    path = geotiff.write_cog(written, tmp_path / "labels.tif")

    with rasterio.open(path) as src:
        assert src.colormap(1)[1] == (0, 255, 0, 255)
        # GDAL reads a palette band as palette-interpreted, whatever else it held.
        assert src.colorinterp == (ColorInterp.palette,)


def test_writing_refuses_a_band_count_the_header_does_not_describe(
    tmp_path: Path,
) -> None:
    # A COG refuses reopening for update, so this writes onto the plain GeoTIFF.
    path = geotiff.write_gtiff(build_raster(), tmp_path / "scene.tif")

    with rasterio.open(path, "r+") as dst, pytest.raises(ValueError, match="2 bands"):
        geotiff.write_header(dst, AttrsHeader())


def test_a_legend_keeps_its_class_names_through_a_geotiff(tmp_path: Path) -> None:
    labelled = build_raster()[["red"]].astype("uint8")
    written = rebase(
        labelled,
        Legend(class_map={0: "bg", 1: "palm"}),
        Legend(color_map={0: "#000000", 1: "#00ff00"}),
        target="red",
    )

    restored = gdal.read(geotiff.write_cog(written, tmp_path / "labels.tif"))

    # A tag holds text, so the class map stays behind but its CF flags travel.
    flags = restored.gs.attrs.data_vars["red"].get(Legend)
    assert flags.flag_values == [0, 1]
    assert flags.class_map == {0: "bg", 1: "palm"}


def test_the_writer_never_states_a_native_value_twice(tmp_path: Path) -> None:
    written = rebase(build_raster(), Packing(scale_factor=1e-4), target=["red", "nir"])

    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    # GDAL holds the scale itself; a metadata item repeating it would say it twice.
    with rasterio.open(path) as src:
        assert src.scales == (1e-4, 1e-4)
        assert "scale_factor" not in src.tags(1)


def test_a_gcp_grid_refuses_to_write_rather_than_lose_its_placing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gcp.tif"
    corners = [
        GroundControlPoint(row=0, col=0, x=3e5, y=5e6),
        GroundControlPoint(row=8, col=8, x=3e5 + 80, y=5e6 - 80),
        GroundControlPoint(row=0, col=8, x=3e5 + 80, y=5e6),
    ]
    with rasterio.open(
        source, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint8"
    ) as dst:
        dst.write(np.ones((1, 8, 8), "uint8"))
        dst.gcps = (corners, CRS.from_epsg(32633))

    # A GeoTIFF records this as control points, which this writer does not write.
    with pytest.raises(ValueError, match="GCPGeoBox"):
        geotiff.write_gtiff(gdal.read(source), tmp_path / "out.tif")


def test_absence_kept_in_a_mask_band_says_it_is_unread(tmp_path: Path) -> None:
    source = tmp_path / "masked.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(3e5, 5e6, 10, 10),
    ) as dst:
        dst.write(np.arange(16, dtype="uint8").reshape(1, 4, 4))
        valid = np.full((4, 4), 255, "uint8")
        valid[0, :] = 0
        dst.write_mask(valid)

    # The mask is not one of the bands read, so its absence would vanish silently.
    with pytest.warns(UnreadMaskWarning, match="mask band"):
        gdal.read(source)


def test_a_files_own_statistics_are_not_carried_into_memory(tmp_path: Path) -> None:
    path = geotiff.write_cog(build_raster(), tmp_path / "scene.tif")
    subprocess.run(["gdalinfo", "-stats", str(path)], check=True, capture_output=True)

    restored = gdal.read(path)

    # GDAL caches them beside the file; they describe the pixels as they were.
    with rasterio.open(path) as src:
        assert "STATISTICS_MINIMUM" in src.tags(1)
    assert restored.red.gs.statistics().minimum is not None
    assert not [key for key in restored.red.attrs if key.startswith("STATISTICS_")]
