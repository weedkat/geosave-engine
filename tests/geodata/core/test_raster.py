from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo import CRS
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import GeoArray, GeoRaster, GeoStack
from geosave_engine.geodata.attrs import (
    ACDD,
    GDALVariable,
    Nodata,
    Packing,
    StackedAttrs,
    rebase,
)
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.io import gdal

from tests.geodata.conftest import build_raster

UTM = "EPSG:32633"


def test_raster_dimensions_do_not_depend_on_coordinate_order() -> None:
    pixels = np.zeros((2, 3, 4, 5))
    built = raster(
        {"red": (("time", "depth", "y", "x"), pixels)},
        coords={"depth": [1, 2, 3], "time": [0, 1]},
    )

    assert built.red.dims == ("time", "depth", "y", "x")
    assert built.red.data is pixels


def test_raster_explicit_dimensions_allow_unlabelled_and_static_axes() -> None:
    built = raster(
        {
            "red": (("time", "y", "x"), np.zeros((2, 4, 5))),
            "dem": (("y", "x"), np.ones((4, 5))),
        }
    )

    assert built.red.dims == ("time", "y", "x")
    assert built.dem.dims == ("y", "x")
    assert "time" not in built.coords


def test_raster_requires_dimensions_for_bare_arrays() -> None:
    with pytest.raises(TypeError, match="dimensions"):
        raster({"red": np.zeros((4, 5))})


def test_raster_accepts_native_dataarrays_without_computing() -> None:
    import dask.array as da

    band = xr.DataArray(da.ones((4, 5), chunks=(2, 5)), dims=("y", "x"))
    built = raster({"red": band})

    assert built.red.data is band.data


@pytest.mark.parametrize(
    ("times", "split_bands", "count"),
    [
        (0, False, 1),
        (0, True, 2),
        (1, False, 1),
        (1, True, 2),
        (2, False, 2),
        (2, True, 4),
    ],
)
def test_cog_export_returns_readable_files(
    tmp_path: Path, times: int, split_bands: bool, count: int
) -> None:
    written = build_raster(times=times)

    paths = written.gs.to_cog(tmp_path / "scene.v2", split_bands=split_bands)

    assert isinstance(paths, tuple)
    assert len(paths) == count
    for path in paths:
        with gdal.read(path) as restored:
            expected = written.sel(time=restored.time) if times else written
            assert restored.gs.geobox == written.gs.geobox
            assert len(restored.data_vars) == (1 if split_bands else 2)
            for name in restored.data_vars:
                np.testing.assert_array_equal(restored[name], expected[name])


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".TIF"])
def test_cog_export_refuses_a_path_that_names_a_tiff(
    tmp_path: Path, suffix: str
) -> None:
    destination = tmp_path / f"scene{suffix}"

    with pytest.raises(ValueError, match="io.geotiff.write_cog"):
        build_raster().gs.to_cog(destination)
    assert not destination.exists()


def test_to_cog_takes_no_id() -> None:
    assert "id" not in inspect.signature(GeoRaster.to_cog).parameters


def geobox(*, crs: str = UTM, shape: tuple[int, int] = (8, 8)) -> GeoBox:
    """Build one grid for a raster-construction test.

    Args:
        crs: CRS the grid sits in.
        shape: Grid height and width in pixels.

    Returns:
        Grid of `shape` at ten-unit pixels.
    """
    left, bottom = 300_000.0, 5_000_000.0
    return GeoBox.from_bbox(
        (left, bottom, left + shape[1] * 10, bottom + shape[0] * 10),
        crs=crs,
        shape=shape,
        tight=True,
    )


def test_raster_places_arrays_and_conforms_them() -> None:
    box = geobox()

    built = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box)

    assert built.gs.geobox == box
    assert built.gs.variables == ("red",)
    assert built.y.attrs["standard_name"] == "projection_y_coordinate"


def test_raster_names_leading_dims_and_labels_them() -> None:
    box = geobox()
    times = pd.date_range("2024-01-01", periods=2, freq="MS")

    built = raster(
        {
            "red": (
                ("time", *("y", "x")),
                np.zeros((2, *box.shape), "uint16"),
            )
        },
        box,
        coords={"time": times},
    )

    assert built.red.dims == ("time", "y", "x")
    assert built.gs.timespan is not None


def test_raster_leaves_an_unlabelled_leading_dim_alone() -> None:
    box = geobox()

    built = raster(
        {
            "probability": (
                ("class", *("y", "x")),
                np.zeros((3, *box.shape), "float32"),
            )
        },
        box,
    )

    assert built.probability.dims == ("class", "y", "x")
    assert "class" not in built.coords


def test_raster_uses_y_x_for_a_geographic_grid() -> None:
    built = raster(
        {
            "red": (
                ("y", "x"),
                np.zeros((4, 4), "uint16"),
            )
        },
        geobox(crs="EPSG:4326", shape=(4, 4)),
    )

    assert built.red.dims == ("y", "x")


def test_raster_without_a_geobox_stays_unplaced() -> None:
    built = raster(
        {
            "band": (
                ("y", "x"),
                np.zeros((4, 4), "uint8"),
            )
        }
    )

    assert built.band.dims == ("y", "x")
    assert not built.coords
    assert built.gs.geobox is None
    assert built.gs.crs_name is None


def test_crs_name_stays_short_however_the_crs_was_built() -> None:
    coded = raster({"red": (("y", "x"), np.zeros((8, 8), "uint16"))}, geobox())
    named = raster(
        {
            "red": (
                ("y", "x"),
                np.zeros((8, 8), "uint16"),
            )
        },
        geobox(crs="+proj=laea +lat_0=52 +lon_0=10"),
    )

    assert coded.gs.crs_name == "EPSG:32633"
    assert coded.red.gs.crs_name == "EPSG:32633"
    # A CRS read back off spatial_ref prints its whole WKT, which this refuses to.
    assert len(str(coded.gs.crs)) > 100
    assert named.gs.crs_name == "unknown"


def test_write_nodata_declares_it_under_both_names() -> None:
    box = geobox()

    built = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box, nodata=0)

    assert built.gs.attrs.data_vars["red"].get(Nodata).fill_value == 0
    assert built.red.odc.nodata == 0


def test_raster_states_no_fill_when_none_is_given() -> None:
    box = geobox()

    built = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box)

    assert "nodata" not in built.red.attrs
    assert "_FillValue" not in built.red.attrs


def test_raster_keeps_pixels_untouched() -> None:
    box = geobox()
    values = np.arange(64, dtype="float32").reshape(box.shape)

    built = raster({"red": (("y", "x"), values)}, box)

    assert np.array_equal(built.red.values, values)
    assert built.red.dtype == np.dtype("float32")


def test_to_array_drops_per_band_gdal_identity() -> None:
    box = geobox()
    built = raster(
        {
            "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B02": (("y", "x"), np.zeros(box.shape, "uint16")),
        },
        box,
    )
    for name, colour in (("B04", "red"), ("B03", "green"), ("B02", "blue")):
        built = rebase(
            built,
            GDALVariable(variable_name=name, colorinterp=colour),
            Packing(scale_factor=1e-4),
            target=name,
        )

    stacked = built.gs.to_array()

    assert stacked.band.values.tolist() == ["B04", "B03", "B02"]
    assert stacked.attrs["scale_factor"] == pytest.approx(1e-4)
    assert "variable_name" not in stacked.attrs
    assert "colorinterp" not in stacked.attrs
    band_attrs = stacked.gs.attrs.coords["band"].get(StackedAttrs)
    assert band_attrs is not None
    assert band_attrs.variable_attrs["B04"]["colorinterp"] == "red"


def test_to_raster_restores_what_to_array_stacked() -> None:
    box = geobox()
    built = rebase(
        raster(
            {
                "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
                "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
            },
            box,
        ),
        ACDD(title="Sentinel-2 Level-2A"),
    )
    built = rebase(
        built,
        GDALVariable(variable_name="B04"),
        Packing(scale_factor=1e-4),
        target="B04",
    )
    built = rebase(
        built,
        GDALVariable(variable_name="B03"),
        Packing(scale_factor=2e-4),
        target="B03",
    )

    split = built.gs.to_array().gs.to_raster()

    assert split.attrs == built.attrs
    for name in ("B04", "B03"):
        assert split[name].attrs == built[name].attrs


def test_to_raster_restores_only_the_bands_still_carried() -> None:
    box = geobox()
    built = raster(
        {
            "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
        },
        box,
    )
    built = rebase(built, GDALVariable(variable_name="B04"), target="B04")
    built = rebase(built, GDALVariable(variable_name="B03"), target="B03")

    split = built.gs.to_array().sel(band=["B04"]).gs.to_raster()

    assert split.gs.variables == ("B04",)
    assert split["B04"].attrs["variable_name"] == "B04"


def test_band_round_trip_keeps_dataset_and_band_attrs_apart() -> None:
    built = raster(
        {"red": (("y", "x"), np.ones(geobox().shape, "float32"))},
        geobox(),
    )
    built.attrs = {"units": "root-unit", "title": "original"}
    built.red.attrs = {"units": "band-unit"}

    stacked = built.gs.to_array()
    assert stacked.attrs == {"units": "band-unit"}
    stacked.attrs["title"] = "edited"
    restored = stacked.gs.to_raster()

    assert restored.attrs == {"units": "root-unit", "title": "original"}
    assert restored.red.attrs == {"units": "band-unit", "title": "edited"}


def test_band_stacking_merges_equivalent_nodata_spellings():
    source = raster(
        {
            "red": (("y", "x"), np.ones(geobox().shape, "int16")),
            "nir": (("y", "x"), np.ones(geobox().shape, "int16")),
        },
        geobox(),
    )
    source.attrs = {"title": "scene", "nodata": -9999}
    source.red.attrs = {"nodata": 0, "long_name": "red"}
    source.nir.attrs = {"_FillValue": 0, "long_name": "nir"}

    stacked = source.gs.to_array()

    assert stacked.attrs == {"nodata": 0, "_FillValue": 0}
    assert stacked.band.attrs["dataset_attrs"] == source.attrs
    assert stacked.band.attrs["variable_attrs"] == {
        "red": {"long_name": "red"},
        "nir": {"long_name": "nir"},
    }
    restored = stacked.sel(band=["nir"]).gs.to_raster()
    assert restored.attrs == source.attrs
    assert list(restored.data_vars) == ["nir"]
    assert restored.nir.attrs == {"long_name": "nir", "nodata": 0, "_FillValue": 0}


def test_statistics_reads_a_lazy_source_once() -> None:
    import dask.array as da
    from dask import delayed

    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([[1.0, 2.0], [3.0, 0.0]])

    values = da.from_delayed(pixels(), shape=(2, 2), dtype="float64")
    band = xr.DataArray(values, dims=("y", "x"), name="red", attrs={"nodata": 0})
    summary = band.gs.statistics()

    assert isinstance(summary, pd.DataFrame)
    assert summary.loc["red", "minimum"] == 1.0
    assert summary.loc["red", "maximum"] == 3.0
    assert summary.loc["red", "mean"] == 2.0
    assert summary.loc["red", "stddev"] == pytest.approx(np.sqrt(2 / 3))
    assert summary.loc["red", "valid_percent"] == 75.0
    assert len(reads) == 1


def test_to_raster_names_an_unstacked_band_after_itself() -> None:
    box = geobox()
    built = rebase(
        raster({"B04": (("y", "x"), np.zeros(box.shape, "uint16"))}, box),
        Packing(scale_factor=1e-4),
        target="B04",
    )

    split = built["B04"].gs.to_raster()

    assert split.gs.variables == ("B04",)
    assert split["B04"].attrs["scale_factor"] == pytest.approx(1e-4)


def test_to_raster_refuses_bands_it_cannot_name() -> None:
    box = geobox()
    stacked = raster(
        {"B04": (("y", "x"), np.zeros(box.shape, "uint16"))}, box
    ).gs.to_array()

    with pytest.raises(ValueError, match="labels none of them"):
        stacked.drop_vars("band").gs.to_raster()


def test_raster_refuses_no_arrays() -> None:
    with pytest.raises(ValueError, match="at least one named array"):
        raster({})


def test_raster_refuses_arrays_on_different_shapes() -> None:
    with pytest.raises(ValueError, match="conflicting sizes"):
        raster(
            {
                "a": (
                    ("y", "x"),
                    np.zeros((8, 8)),
                ),
                "b": (
                    ("y", "x"),
                    np.zeros((4, 4)),
                ),
            }
        )


def test_raster_refuses_a_rank_that_does_not_match_dims() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="Could not convert tuple"):
        raster({"a": (("y", "x"), np.zeros((3, *box.shape)))}, box)


def test_raster_refuses_trailing_axes_off_the_grid() -> None:
    with pytest.raises(ValueError, match="geobox"):
        raster({"a": (("y", "x"), np.zeros((4, 4)))}, geobox())


def test_raster_refuses_a_coord_of_the_wrong_length() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="conflicting sizes"):
        raster(
            {
                "a": (
                    ("class", *("y", "x")),
                    np.zeros((3, *box.shape)),
                )
            },
            box,
            coords={"class": [1, 2]},
        )


def test_statistics_summarise_only_the_present_pixels() -> None:
    box = geobox()
    pixels = np.arange(64, dtype="uint16").reshape(box.shape)
    band = raster({"red": (("y", "x"), pixels)}, box, nodata=0)["red"]

    summary = band.gs.statistics()

    # GDAL summarises the present pixels, so 0 is absence rather than a reading.
    assert summary.loc["red", "minimum"] == 1.0
    assert summary.loc["red", "mean"] == pytest.approx(32.0)
    assert summary.loc["red", "valid_percent"] == pytest.approx(98.4375)


def test_nodata_reads_back_what_was_written() -> None:
    box = geobox()
    pixels = np.zeros(box.shape, "uint16")
    built = raster(
        {"red": (("y", "x"), pixels), "nir": (("y", "x"), pixels)},
        box,
    ).gs.write_nodata(0, target="red")

    assert built.red.gs.nodata == 0
    assert built.nir.gs.nodata is None
    assert built.gs.nodata == {"red": 0, "nir": None}


def test_raster_statistics_tabulate_each_variable_as_its_band_does() -> None:
    box = geobox()
    pixels = np.arange(64, dtype="uint16").reshape(box.shape)
    built = raster(
        {
            "red": (("y", "x"), pixels),
            "nir": (("y", "x"), pixels * 2),
        },
        box,
        nodata=0,
    )

    table = built.gs.statistics()

    assert list(table.index) == ["red", "nir"]
    assert list(table.columns) == [
        "minimum",
        "maximum",
        "mean",
        "stddev",
        "valid_percent",
    ]
    pd.testing.assert_frame_equal(table.loc[["nir"]], built.nir.gs.statistics())


def test_multiband_array_statistics_preserve_band_order_and_nodata() -> None:
    source = xr.Dataset(
        {
            "nir": (("y", "x"), [[0.0, 10.0, 20.0]]),
            "red": (("y", "x"), [[-9999.0, 1.0, 3.0]]),
        }
    ).gs.rebase(Nodata(fill_value=0), target="nir")
    source = source.gs.rebase(Nodata(fill_value=-9999), target="red")

    table = source.gs.to_array().gs.statistics()

    pd.testing.assert_frame_equal(table, source.gs.statistics())
    assert list(table.index) == ["nir", "red"]
    assert table.loc["nir", "mean"] == 15.0
    assert table.loc["red", "mean"] == 2.0
    assert table.loc["red", "valid_percent"] == pytest.approx(200 / 3)


@pytest.mark.parametrize("name", [None, "red"])
def test_single_array_statistics_use_the_array_name(name) -> None:
    band = xr.DataArray([1.0, np.nan, 3.0], dims="x", name=name)

    table = band.gs.statistics()

    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == [name]
    assert table.iloc[0]["mean"] == 2.0
    assert table.iloc[0]["valid_percent"] == pytest.approx(200 / 3)


def test_dataset_statistics_have_one_row_per_variable_with_band_dimensions() -> None:
    source = xr.Dataset(
        {"image": (("band", "x"), [[1, 3], [10, 20]])},
        coords={"band": ["red", "nir"]},
    )

    table = source.gs.statistics()

    assert list(table.index) == ["image"]
    assert table.loc["image", "mean"] == 8.5


def test_multiband_statistics_refuse_an_absent_band() -> None:
    image = xr.DataArray(
        [[1.0, 3.0], [np.nan, np.nan]],
        dims=("band", "x"),
        coords={"band": ["red", "nir"]},
    )

    with pytest.raises(ValueError, match="nir holds no present pixel"):
        image.gs.statistics()


def test_multiband_statistics_require_band_labels() -> None:
    image = xr.DataArray([[1, 3], [10, 20]], dims=("band", "x"))

    with pytest.raises(ValueError, match="labels none"):
        image.gs.statistics()


def test_statistics_refuse_a_band_holding_no_present_pixel() -> None:
    box = geobox()
    band = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box, nodata=0)[
        "red"
    ]

    with pytest.raises(ValueError, match="no present pixel"):
        band.gs.statistics()


def test_band_round_trip_keeps_dataset_root_nodata_as_a_foreign_attr():
    source = raster(
        {"red": (("y", "x"), np.ones(geobox().shape, "uint16"))},
        geobox(),
    )
    source.attrs = {"nodata": -9999}
    source.red.attrs = {"nodata": 0}
    restored = source.gs.to_array().gs.to_raster()
    assert restored.gs.attrs.root.get(Nodata) is None
    assert restored.attrs == {"nodata": -9999}
    assert restored.red.gs.attrs.root.get(Nodata).fill_value == 0


def test_band_stacking_stores_root_values_in_their_json_spelling():
    from pathlib import Path

    source = raster({"red": (("y", "x"), np.ones(geobox().shape))}, geobox())
    source.attrs = {"source_path": Path("local.tif")}
    assert source.gs.to_array().gs.to_raster().attrs == {"source_path": "local.tif"}


def test_an_edit_on_the_stacked_array_reaches_every_band():
    source = raster(
        {
            "red": (("y", "x"), np.ones(geobox().shape, "int16")),
            "nir": (("y", "x"), np.ones(geobox().shape, "int16")),
        },
        geobox(),
    )
    source.red.attrs = {"nodata": 0, "units": "1"}
    source.nir.attrs = {"nodata": 0, "units": "1"}
    stacked = rebase(source.gs.to_array(), Nodata(fill_value=-1))
    restored = stacked.gs.to_raster()
    assert restored.red.gs.attrs.root.get(Nodata).fill_value == -1
    assert restored.nir.gs.attrs.root.get(Nodata).fill_value == -1


def test_bands_with_different_nodata_round_trip():
    source = raster(
        {
            "red": (("y", "x"), np.ones(geobox().shape, "int16")),
            "nir": (("y", "x"), np.ones(geobox().shape, "int16")),
        },
        geobox(),
    )
    source.red.attrs = {"nodata": 0, "units": "1"}
    source.nir.attrs = {"nodata": -1, "units": "1"}
    source.attrs = {"title": "S2"}
    stacked = source.gs.to_array()
    assert stacked.attrs == {"units": "1"}
    restored = stacked.gs.to_raster()
    assert restored.red.attrs == {"nodata": 0, "_FillValue": 0, "units": "1"}
    assert restored.nir.attrs == {"nodata": -1, "_FillValue": -1, "units": "1"}
    assert restored.attrs == {"title": "S2"}


def test_an_array_without_stacked_attrs_gives_its_attrs_to_every_band():
    array = xr.DataArray(
        np.zeros((2, 2, 2), "int16"),
        dims=("band", "y", "x"),
        coords={"band": ["red", "nir"]},
        attrs={"nodata": 0},
    )
    restored = array.gs.to_raster()
    assert restored.attrs == {}
    assert restored.red.attrs == {"nodata": 0, "_FillValue": 0}
    assert restored.nir.attrs == {"nodata": 0, "_FillValue": 0}


def test_stacking_preserves_nan_nodata_at_each_scope():
    source = raster(
        {"red": (("y", "x"), np.ones(geobox().shape, "float32"))},
        geobox(),
    )
    source.attrs = {"nodata": np.nan}
    source.red.attrs = {"nodata": np.float32("nan")}
    stacked = source.gs.to_array()
    assert np.isnan(stacked.gs.attrs.root.get(Nodata).fill_value)
    restored = stacked.gs.to_raster()
    assert np.isnan(restored.attrs["nodata"])
    assert np.isnan(restored.red.gs.attrs.root.get(Nodata).fill_value)


def test_write_crs_replaces_grid_coordinate_attrs_and_keeps_the_root() -> None:
    built = raster({"red": (("y", "x"), np.ones(geobox().shape))}, geobox())
    built.attrs = {"title": "S2"}
    built.x.attrs["crs"] = "EPSG:4326"

    written = built.gs.write_crs()

    assert written.attrs == {"title": "S2"}
    assert CRS(written.x.attrs["crs"]) == CRS(UTM)
    assert written.x.attrs["axis"] == "X"


def test_write_crs_keeps_a_one_pixel_grid() -> None:
    built = raster(
        {"red": (("y", "x"), np.ones((1, 1)))},
        geobox(shape=(1, 1)),
    )

    assert built.gs.write_crs().odc.geobox == built.odc.geobox


def test_array_builds_the_band_raster_would() -> None:
    from geosave_engine.geodata.core.array import array

    pixels = np.ones((2, *geobox().shape), "int16")
    labels = pd.date_range("2024-01-01", periods=2)

    band = array(
        pixels,
        geobox(),
        dims=("time", *("y", "x")),
        coords={"time": labels},
        nodata=0,
    )
    variable = raster(
        {
            "a": (
                ("time", *("y", "x")),
                pixels,
            )
        },
        geobox(),
        coords={"time": labels},
        nodata=0,
    )["a"]

    assert band.name is None
    xr.testing.assert_identical(band, variable.rename(None))


@pytest.mark.parametrize(
    ("pixels", "dims", "coords", "message"),
    [
        (np.ones((8, 8)), ("time", "y", "x"), {"time": [0]}, "Could not convert tuple"),
        (np.ones((2, 8, 8)), ("time", "y", "x"), {"time": [0]}, "conflicting sizes"),
        (np.ones((4, 4)), ("y", "x"), {}, "geobox"),
        (np.ones((8, 8)), ("y", "x"), {"x": range(8)}, "already supplies"),
    ],
)
def test_array_refuses_what_raster_refuses(pixels, dims, coords, message) -> None:
    from geosave_engine.geodata.core.array import array

    with pytest.raises(ValueError, match=message):
        array(
            pixels,
            geobox(),
            dims=dims,
            coords=coords,
        )


@pytest.mark.parametrize(
    "writer",
    [
        GeoRaster.to_cog,
        GeoRaster.to_zarr,
        GeoRaster.to_netcdf,
        GeoStack.to_cog,
        GeoStack.to_zarr,
        GeoStack.to_netcdf,
        GeoArray.to_cog,
    ],
)
def test_writers_take_no_catalog_argument(writer) -> None:
    assert "catalog" not in inspect.signature(writer).parameters


@pytest.mark.parametrize(
    "writer",
    [
        GeoRaster.to_zarr,
        GeoRaster.to_netcdf,
        GeoStack.to_cog,
        GeoStack.to_zarr,
        GeoStack.to_netcdf,
        GeoArray.to_cog,
    ],
)
def test_writers_that_name_nothing_take_no_id(writer) -> None:
    assert "id" not in inspect.signature(writer).parameters


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857"])
def test_constructor_spatial_names_preserve_cf_grid_through_netcdf(
    tmp_path: Path, crs: str
) -> None:
    import dask.array as da
    from affine import Affine

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), crs)
    pixels = da.ones((2, 3), chunks=(1, 3))
    built = raster({"red": (("y", "x"), pixels)}, grid)

    assert built.red.data is pixels
    assert built.red.dims == ("y", "x")
    assert built.gs.write_crs().red.dims == ("y", "x")
    assert built.y.attrs["axis"] == "Y"
    assert built.x.attrs["axis"] == "X"
    if crs == "EPSG:4326":
        assert built.y.attrs["standard_name"] == "latitude"
        assert built.x.attrs["standard_name"] == "longitude"
        assert built.y.attrs["units"] == "degrees_north"
        assert built.x.attrs["units"] == "degrees_east"
    else:
        assert built.y.attrs["standard_name"] == "projection_y_coordinate"
        assert built.x.attrs["standard_name"] == "projection_x_coordinate"

    path = tmp_path / "grid.nc"
    built.to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as restored:
        assert restored.red.dims == ("y", "x")
        assert restored.odc.geobox == grid
        assert restored.rio.crs.to_epsg() == int(crs.split(":")[1])
        assert (restored.rio.y_dim, restored.rio.x_dim) == ("y", "x")
        xr.testing.assert_equal(restored.red, built.red.compute())


def test_a_band_indexes_as_its_one_variable_raster(tmp_path: Path) -> None:
    from geosave_engine.geodata.stac import create_items

    band = build_raster(times=2)["red"].isel(time=0)
    path = band.gs.to_cog(tmp_path / "red.tif")

    (item,) = create_items(path)

    assert item.id == "red"
    assert list(item.assets) == ["red"]
    assert item.assets["red"].href == str(path)
