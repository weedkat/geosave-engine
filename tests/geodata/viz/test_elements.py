"""Rendering one georeferenced DataArray through holoviews.

One description, two backends: `to_element` builds what bokeh shows
interactively, `plot` renders the same thing to a matplotlib Figure. The
invariants here are which renderer a set of arguments selects, that nodata
never reaches the color range, and that a datashaded view reads no pixels.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.xr import xr_coords

hv = pytest.importorskip("holoviews", reason="needs the viz extra")
pytest.importorskip("hvplot", reason="needs the viz extra")

from geosave_engine.geodata.extensions import RenderHints  # noqa: E402
from geosave_engine.geodata.viz import (  # noqa: E402
    RenderStyle,
    plot,
    to_element,
    to_static_element,
)

from ..spatial.conftest import make_anchor, make_raster  # noqa: E402


def array(bands=("B04",), *, values=None, nodata=np.nan, width=16, height=16):
    """Plain georeferenced DataArray, no GeoTile/GeoRaster wrapper."""
    anchor = make_anchor(width=width, height=height)
    pixels = np.ones((len(bands), height, width), dtype="float32") if values is None else values
    da = xr.DataArray(
        pixels,
        dims=("band", "y", "x"),
        coords={"band": list(bands), **dict(xr_coords(anchor.geobox, always_yx=True))},
    )
    return da.rio.write_nodata(nodata)


def style_of(element):
    """Style options holoviews resolved for an element."""
    return element.opts.get("style").kwargs


class TestRendererSelection:
    """Which renderer applies follows from arguments and band count, never from pixel values."""

    def test_three_bands_compose_as_rgb(self):
        assert isinstance(to_element(array(("B04", "B03", "B02"))), hv.RGB)

    def test_named_rgb_bands_compose_from_a_wider_array(self):
        assert isinstance(to_element(array(("B02", "B03", "B04", "B08")), render=RenderHints(rgb_bands=("B04", "B03", "B02"))), hv.RGB)

    def test_four_bands_without_rgb_bands_do_not_guess(self):
        assert not isinstance(to_element(array(("B02", "B03", "B04", "B08"))), hv.RGB)

    def test_rgb_bands_naming_a_missing_band_raises(self):
        with pytest.raises(ValueError, match="must name three of this array's bands"):
            to_element(array(("B04", "B03", "B02")), render=RenderHints(rgb_bands=("B04", "B03", "nope")))

    def test_rgb_without_a_band_dim_raises(self):
        with pytest.raises(ValueError, match="needs a band dim"):
            to_element(array(("B04",)).isel(band=0, drop=True), render=RenderHints(rgb_bands=("a", "b", "c")))

    def test_color_map_drives_a_discrete_palette(self):
        assert style_of(to_element(array(), render=RenderHints(color_map={0: "#FFFFFF", 1: "#000000"})))["cmap"] == ["#FFFFFF", "#000000"]

    def test_a_categorical_layer_is_never_datashaded(self):
        """Interpolating between class values would invent classes that don't exist."""
        element = to_element(array(), render=RenderHints(class_map={0: "clear", 1: "cloud"}), rasterize=True)
        assert not isinstance(element, hv.DynamicMap)

    def test_class_labels_become_a_legend(self):
        """A colorbar's tick labels are backend-specific, so classes are named by an overlay."""
        element = to_element(array(), render=RenderHints(class_map={0: "clear", 1: "cloud"}))
        legend = next(child for child in element if isinstance(child, hv.NdOverlay))
        assert [str(key) for key in legend.keys()] == ["clear", "cloud"]


class TestColorRange:
    def test_values_straddling_zero_pick_a_diverging_colormap(self):
        spread = np.linspace(-1, 1, 256, dtype="float32").reshape(1, 16, 16)
        assert style_of(to_element(array(values=spread), rasterize=False))["cmap"] == "RdBu_r"

    def test_positive_values_stay_sequential(self):
        positive = np.linspace(0.1, 1, 256, dtype="float32").reshape(1, 16, 16)
        assert style_of(to_element(array(values=positive), rasterize=False))["cmap"] == "viridis"

    def test_an_explicit_colormap_wins(self):
        spread = np.linspace(-1, 1, 256, dtype="float32").reshape(1, 16, 16)
        assert style_of(to_element(array(values=spread), cmap="magma", rasterize=False))["cmap"] == "magma"

    def test_outliers_are_clipped_by_the_percentile_stretch(self):
        values = np.ones((1, 16, 16), dtype="float32")
        values[0, 0, 0] = 1000.0
        low, high = to_element(array(values=values), rasterize=False).range("value")
        assert high < 1000.0

    def test_constant_data_gets_a_usable_range(self):
        """Equal low and high would collapse the colorbar."""
        low, high = to_element(array(), rasterize=False).range("value")
        assert high > low


class TestPolicy:
    """Stretch and colormaps are policy a caller can replace, not constants baked into the renderer."""

    def test_style_replaces_the_default_colormaps(self):
        positive = np.linspace(0.1, 1, 256, dtype="float32").reshape(1, 16, 16)
        style = RenderStyle(sequential_cmap="magma")
        assert style_of(to_element(array(values=positive), style=style, rasterize=False))["cmap"] == "magma"

    def test_style_replaces_the_stretch_percentiles(self):
        spread = np.linspace(0, 100, 256, dtype="float32").reshape(1, 16, 16)
        wide = to_element(array(values=spread), style=RenderStyle(stretch=(0.0, 100.0)), rasterize=False)
        narrow = to_element(array(values=spread), style=RenderStyle(stretch=(25.0, 75.0)), rasterize=False)
        assert wide.range("value")[1] > narrow.range("value")[1]

    def test_an_ill_formed_stretch_is_rejected(self):
        with pytest.raises(ValueError, match="ascending pair inside 0–100"):
            RenderStyle(stretch=(98.0, 2.0))

    def test_kind_forces_a_renderer_against_the_hints(self):
        """A three-band array normally composes as RGB; kind overrides that."""
        assert not isinstance(to_element(array(("B04", "B03", "B02")), kind="continuous"), hv.RGB)

    def test_kind_forces_rgb_without_hints(self):
        assert isinstance(to_element(array(("a", "b", "c", "d")).isel(band=slice(0, 3)), kind="rgb"), hv.RGB)


class TestNodata:
    def test_nodata_is_masked_out_of_the_pixels(self):
        values = np.ones((1, 16, 16), dtype="float32")
        values[0, 0, 0] = -9999
        element = to_element(array(values=values, nodata=-9999), rasterize=False)
        assert np.isnan(element.data["value"].values).any()

    def test_nodata_never_reaches_the_color_range(self):
        values = np.ones((1, 16, 16), dtype="float32")
        values[0, 0, 0] = -9999
        low, _ = to_element(array(values=values, nodata=-9999), rasterize=False).range("value")
        assert low > -9999

    def test_an_all_nodata_array_still_renders(self):
        values = np.full((1, 16, 16), -9999, dtype="float32")
        assert to_element(array(values=values, nodata=-9999), rasterize=False) is not None


class TestLaziness:
    def test_a_datashaded_view_reads_no_pixels(self):
        """The whole point of the unbounded path — building the view must not touch the surface."""
        import dask.array as darr

        def explode(block):
            raise AssertionError("pixels were read while building the view")

        anchor = make_anchor(width=64, height=64)
        pixels = darr.map_blocks(explode, darr.zeros((64, 64), chunks=32), dtype="float32")
        da = xr.DataArray(pixels, dims=("y", "x"), coords=dict(xr_coords(anchor.geobox, always_yx=True)))
        assert to_element(da.rio.write_nodata(np.nan), rasterize=True) is not None

    def test_an_unrasterized_view_is_a_plain_element(self):
        assert isinstance(to_element(make_raster(bands=("B04",)).data, rasterize=False), hv.Image)


class TestStaticPanel:
    def test_a_single_step_squeezes_instead_of_becoming_a_widget(self):
        """A length-1 dim would render as an animation rather than a figure."""
        assert not isinstance(to_static_element(array()), hv.DynamicMap)

    def test_more_than_one_step_raises(self):
        series = make_raster(
            bands=("B04",),
            times=[np.datetime64("2024-01-15"), np.datetime64("2024-02-15")],
            time=("2024-01-01", "2024-02-29"),
        )
        with pytest.raises(ValueError, match="pass band=/time= to pick one"):
            plot(series.data)

    def test_plot_returns_a_matplotlib_figure(self):
        from matplotlib.figure import Figure

        assert isinstance(plot(array()), Figure)

    def test_three_bands_stay_one_rgb_panel(self):
        from matplotlib.figure import Figure

        assert isinstance(plot(array(("B04", "B03", "B02"))), Figure)
