from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs import (
    ACDD,
    GDALVariable,
    Nodata,
    Packing,
    StackedAttrs,
    rebase,
)
from geosave_engine.geodata.core.raster import raster

UTM = "EPSG:32633"


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

    built = raster({"red": np.zeros(box.shape, "uint16")}, box)

    assert built.gs.geobox == box
    assert built.gs.variables == ("red",)
    assert built.y.attrs["standard_name"] == "projection_y_coordinate"


def test_raster_names_leading_dims_and_labels_them() -> None:
    box = geobox()
    times = pd.date_range("2024-01-01", periods=2, freq="MS")

    built = raster({"red": np.zeros((2, *box.shape), "uint16")}, box, time=times)

    assert built.red.dims == ("time", *box.dimensions)
    assert built.gs.timespan is not None


def test_raster_leaves_an_unlabelled_leading_dim_alone() -> None:
    box = geobox()

    built = raster(
        {"probability": np.zeros((3, *box.shape), "float32")}, box, **{"class": None}
    )

    assert built.probability.dims == ("class", *box.dimensions)
    assert "class" not in built.coords


def test_raster_takes_spatial_names_from_the_grid() -> None:
    built = raster(
        {"red": np.zeros((4, 4), "uint16")}, geobox(crs="EPSG:4326", shape=(4, 4))
    )

    assert built.gs.grid_dims == ("latitude", "longitude")


def test_raster_without_a_geobox_stays_unplaced() -> None:
    built = raster({"band": np.zeros((4, 4), "uint8")})

    assert built.band.dims == ("y", "x")
    assert not built.coords
    assert built.gs.geobox is None
    assert built.gs.crs_name is None


def test_crs_name_stays_short_however_the_crs_was_built() -> None:
    coded = raster({"red": np.zeros((8, 8), "uint16")}, geobox())
    named = raster(
        {"red": np.zeros((8, 8), "uint16")},
        geobox(crs="+proj=laea +lat_0=52 +lon_0=10"),
    )

    assert coded.gs.crs_name == "EPSG:32633"
    assert coded.red.gs.crs_name == "EPSG:32633"
    # A CRS read back off spatial_ref prints its whole WKT, which this refuses to.
    assert len(str(coded.gs.crs)) > 100
    assert named.gs.crs_name == "unknown"


def test_write_nodata_declares_it_under_both_names() -> None:
    box = geobox()

    built = raster({"red": np.zeros(box.shape, "uint16")}, box, nodata=0)

    assert built.gs.attrs.data_vars["red"].get(Nodata).fill_value == 0
    assert built.red.odc.nodata == 0


def test_raster_states_no_fill_when_none_is_given() -> None:
    box = geobox()

    built = raster({"red": np.zeros(box.shape, "uint16")}, box)

    assert "nodata" not in built.red.attrs
    assert "_FillValue" not in built.red.attrs


def test_raster_keeps_pixels_untouched() -> None:
    box = geobox()
    values = np.arange(64, dtype="float32").reshape(box.shape)

    built = raster({"red": values}, box)

    assert np.array_equal(built.red.values, values)
    assert built.red.dtype == np.dtype("float32")


def test_to_array_drops_per_band_gdal_identity() -> None:
    box = geobox()
    built = raster(
        {
            "B04": np.zeros(box.shape, "uint16"),
            "B03": np.zeros(box.shape, "uint16"),
            "B02": np.zeros(box.shape, "uint16"),
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
    parked = stacked.gs.attrs.coords["band"].get(StackedAttrs)
    assert parked is not None
    assert parked.variable_attrs["B04"]["colorinterp"] == "red"


def test_to_raster_restores_what_to_array_stacked() -> None:
    box = geobox()
    built = rebase(
        raster(
            {
                "B04": np.zeros(box.shape, "uint16"),
                "B03": np.zeros(box.shape, "uint16"),
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
        {"B04": np.zeros(box.shape, "uint16"), "B03": np.zeros(box.shape, "uint16")},
        box,
    )
    built = rebase(built, GDALVariable(variable_name="B04"), target="B04")
    built = rebase(built, GDALVariable(variable_name="B03"), target="B03")

    split = built.gs.to_array().sel(band=["B04"]).gs.to_raster()

    assert split.gs.variables == ("B04",)
    assert split["B04"].attrs["variable_name"] == "B04"


def test_band_round_trip_preserves_overlapping_root_metadata() -> None:
    built = raster({"red": np.ones(geobox().shape, "float32")}, geobox())
    built.attrs = {"units": "root-unit", "title": "original"}
    built.red.attrs = {"units": "band-unit"}

    stacked = built.gs.to_array()
    stacked.attrs["title"] = "edited"
    restored = stacked.gs.to_raster()

    assert restored.attrs == {"units": "root-unit", "title": "edited"}
    assert restored.red.attrs == {"units": "band-unit"}


def test_statistics_reads_a_lazy_source_once() -> None:
    import dask.array as da
    from dask import delayed

    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([[1.0, 2.0], [3.0, 0.0]])

    values = da.from_delayed(pixels(), shape=(2, 2), dtype="float64")
    band = xr.DataArray(values, dims=("y", "x"), attrs={"nodata": 0})
    summary = band.gs.statistics()

    assert summary.minimum == 1.0
    assert summary.maximum == 3.0
    assert summary.mean == 2.0
    assert summary.stddev == pytest.approx(np.sqrt(2 / 3))
    assert summary.valid_percent == 75.0
    assert len(reads) == 1


def test_to_raster_names_an_unstacked_band_after_itself() -> None:
    box = geobox()
    built = rebase(
        raster({"B04": np.zeros(box.shape, "uint16")}, box),
        Packing(scale_factor=1e-4),
        target="B04",
    )

    split = built["B04"].gs.to_raster()

    assert split.gs.variables == ("B04",)
    assert split["B04"].attrs["scale_factor"] == pytest.approx(1e-4)


def test_to_raster_refuses_bands_it_cannot_name() -> None:
    box = geobox()
    stacked = raster({"B04": np.zeros(box.shape, "uint16")}, box).gs.to_array()

    with pytest.raises(ValueError, match="labels none of them"):
        stacked.drop_vars("band").gs.to_raster()


def test_raster_refuses_no_arrays() -> None:
    with pytest.raises(ValueError, match="at least one named array"):
        raster({})


def test_raster_refuses_arrays_on_different_shapes() -> None:
    with pytest.raises(ValueError, match="one axis is one length"):
        raster({"a": np.zeros((8, 8)), "b": np.zeros((4, 4))})


def test_raster_refuses_a_rank_that_does_not_match_dims() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="dimensional"):
        raster({"a": np.zeros((3, *box.shape))}, box)


def test_raster_refuses_trailing_axes_off_the_grid() -> None:
    with pytest.raises(ValueError, match="trailing axes"):
        raster({"a": np.zeros((4, 4))}, geobox())


def test_raster_refuses_a_coord_of_the_wrong_length() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="2 labels for an axis of 3"):
        raster({"a": np.zeros((3, *box.shape))}, box, **{"class": [1, 2]})


def test_statistics_summarise_only_the_present_pixels() -> None:
    box = geobox()
    pixels = np.arange(64, dtype="uint16").reshape(box.shape)
    band = raster({"red": pixels}, box, nodata=0)["red"]

    summary = band.gs.statistics()

    # GDAL summarises the present pixels, so 0 is absence rather than a reading.
    assert summary.minimum == 1.0
    assert summary.mean == pytest.approx(32.0)
    assert summary.valid_percent == pytest.approx(98.4375)


def test_statistics_refuse_a_band_holding_no_present_pixel() -> None:
    box = geobox()
    band = raster({"red": np.zeros(box.shape, "uint16")}, box, nodata=0)["red"]

    with pytest.raises(ValueError, match="no present pixel"):
        band.gs.statistics()


def test_band_round_trip_preserves_root_nodata_aliases():
    source = raster({"red": np.ones(geobox().shape, "uint16")}, geobox())
    source.attrs = {"nodata": -9999}
    source.red.attrs = {"nodata": 0}
    restored = source.gs.to_array().gs.to_raster()
    assert restored.gs.attrs.root.get(Nodata).fill_value == -9999
    assert restored.red.gs.attrs.root.get(Nodata).fill_value == 0


def test_band_stacking_keeps_unrelated_python_root_values():
    from pathlib import Path

    source = raster({"red": np.ones(geobox().shape)}, geobox())
    source.attrs = {"source_path": Path("local.tif")}
    assert source.gs.to_array().gs.to_raster().attrs == source.attrs


def test_band_round_trip_keeps_edits_to_root_keys_not_lifted_from_bands():
    source = raster(
        {"red": np.ones(geobox().shape), "nir": np.ones(geobox().shape)}, geobox()
    )
    source.attrs = {"units": "root"}
    source.red.attrs = {"units": "red-unit"}
    source.nir.attrs = {"units": "nir-unit"}
    stacked = source.gs.to_array()
    stacked.attrs["units"] = "edited-root"
    assert stacked.sel(band=["red"]).gs.to_raster().attrs["units"] == "edited-root"


def test_stacking_preserves_nan_nodata_at_each_scope():
    source = raster({"red": np.ones(geobox().shape, "float32")}, geobox())
    source.attrs = {"nodata": np.nan}
    source.red.attrs = {"nodata": np.float32("nan")}
    stacked = source.gs.to_array()
    assert np.isnan(stacked.gs.attrs.root.get(Nodata).fill_value)
    restored = stacked.gs.to_raster()
    assert np.isnan(restored.gs.attrs.root.get(Nodata).fill_value)
    assert np.isnan(restored.red.gs.attrs.root.get(Nodata).fill_value)
