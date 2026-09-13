import matplotlib
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.raster import raster as build_raster

pytest.importorskip("hvplot", reason="viz extra not installed")

import holoviews as hv  # noqa: E402

matplotlib.use("agg")
hv.extension("matplotlib")


def geobox() -> GeoBox:
    return GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_040.0, 9_000_040.0),
        crs="EPSG:32749",
        shape=(4, 4),
        tight=True,
    )


@pytest.fixture
def optical() -> xr.Dataset:
    values = np.arange(16, dtype="uint16").reshape(4, 4) * 100
    raster = build_raster({"B04": values}, geobox())
    return raster.gs.rebase(
        attrs.Packing(scale_factor=1e-4, add_offset=0.0),
        attrs.Nodata(_FillValue=0),
        target="B04",
    )


@pytest.fixture
def labels() -> xr.Dataset:
    """Class map whose codes are sparse, as a real land-cover product's are."""
    codes = np.array(
        [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
    )
    raster = build_raster({"landcover": codes}, geobox())
    return raster.gs.rebase(
        attrs.Legend(
            class_map={0: "water", 1: "trees", 8: "snow"},
            color_map={0: "#419bdf", 1: "#397d49", 8: "#b39fe1"},
        ),
        target="landcover",
    )


def drawn_values(element) -> np.ndarray:
    return element.dimension_values(2)


class TestPacking:
    def test_draws_stored_values(self, optical: xr.Dataset) -> None:
        assert np.nanmax(drawn_values(optical.gs.plot("B04"))) == 1500

    def test_leaves_the_declared_fill_in_place(self, optical: xr.Dataset) -> None:
        assert not np.isnan(drawn_values(optical.gs.plot("B04"))).any()

    def test_identity_packing_is_left_alone(self, labels: xr.Dataset) -> None:
        packed = labels.gs.rebase(
            attrs.Packing(scale_factor=1.0, add_offset=0.0), target="landcover"
        )
        assert packed.gs.plot("landcover") is not None


class TestClassMap:
    def test_draws_classes_on_position(self, labels: xr.Dataset) -> None:
        assert sorted(set(drawn_values(labels.gs.plot("landcover")))) == [0, 1, 2]

    def test_colours_follow_class_order(self, labels: xr.Dataset) -> None:
        style = (
            labels.gs.plot("landcover").opts.get("style", backend="matplotlib").kwargs
        )
        assert list(style["cmap"].colors) == ["#419bdf", "#397d49", "#b39fe1"]

    def test_names_every_class_on_the_colorbar(self, labels: xr.Dataset) -> None:
        drawn = labels.gs.plot("landcover")
        assert drawn.opts.get("plot", backend="matplotlib").kwargs["cbar_ticks"] == [
            (0, "water"),
            (1, "trees"),
            (2, "snow"),
        ]

    def test_accepts_an_rgb_tuple_palette(self, labels: xr.Dataset) -> None:
        recoloured = labels.gs.rebase(
            attrs.Legend(
                class_map={0: "water", 1: "trees", 8: "snow"},
                color_map={0: (65, 155, 223), 1: (57, 125, 73), 8: (179, 159, 225)},
            ),
            target="landcover",
        )
        style = (
            recoloured.gs.plot("landcover")
            .opts.get("style", backend="matplotlib")
            .kwargs
        )
        assert style["cmap"].colors[0] == "#419bdf"

    def test_absent_pixels_draw_as_no_class(self) -> None:
        codes = np.array(
            [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
        )
        raster = build_raster({"cover": codes}, geobox(), nodata=0)
        raster = raster.gs.rebase(
            attrs.Legend(
                class_map={1: "trees", 8: "snow"},
                color_map={1: "#397d49", 8: "#b39fe1"},
            ),
            target="cover",
        )
        drawn = drawn_values(raster.gs.plot("cover"))
        assert np.isnan(drawn).any()
        assert set(np.unique(drawn[~np.isnan(drawn)])) == {0.0, 1.0}

    def test_uncoloured_class_refuses(self) -> None:
        raster = build_raster({"cover": np.zeros((4, 4), dtype="uint8")}, geobox())
        raster = raster.gs.rebase(
            attrs.Legend(class_map={0: "water", 1: "trees"}, color_map={0: "#419bdf"}),
            target="cover",
        )
        with pytest.raises(ValueError, match="carry no colour"):
            raster.gs.plot("cover")


class TestConstraints:
    def test_unknown_variable_refuses(self, optical: xr.Dataset) -> None:
        with pytest.raises(KeyError, match="not data variables"):
            optical.gs.plot("B99")

    def test_selecting_first_draws(self, optical: xr.Dataset) -> None:
        assert optical.expand_dims(time=3).isel(time=0).gs.plot("B04") is not None


class TestOutput:
    def test_saves_a_png(self, labels: xr.Dataset, tmp_path) -> None:
        written = tmp_path / "landcover.png"
        hv.save(labels.gs.plot("landcover"), written, backend="matplotlib", fmt="png")
        assert written.stat().st_size > 0

    def test_opts_reach_hvplot(self, optical: xr.Dataset) -> None:
        drawn = optical.gs.plot("B04", cmap="RdYlGn")
        assert drawn.opts.get("style").kwargs["cmap"] == "RdYlGn"

    def test_elements_compose(self, optical: xr.Dataset, labels: xr.Dataset) -> None:
        assert isinstance(
            optical.gs.plot("B04") + labels.gs.plot("landcover"), hv.Layout
        )
