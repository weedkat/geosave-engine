"""The canonical Spatial array contract — dims, bands, time, CRS, dtype, nodata.

These are the invariants every other Spatial type is built on, so a break
here is a break everywhere. Grid-preserving transforms belong to
`_SpatialArray` and must keep a `tiling` stamp valid across them.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.spatial import GeoAnchor, GeoRaster, GeoTile

from .conftest import UTM, is_lazy, make_anchor, make_geobox, make_lazy_raster, make_raster


class TestCanonicalRepresentation:
    def test_dims_must_be_band_y_x_or_time_band_y_x(self):
        anchor = make_anchor(width=4, height=4)
        bare = xr.DataArray(
            np.ones((4, 4), dtype="uint16"),
            dims=("y", "x"),
            coords=dict(xr_coords(anchor.geobox, always_yx=True)),
        )
        with pytest.raises(ValueError, match=r"dims \('band', 'y', 'x'\)"):
            anchor.to_raster(bare)

    def test_raw_numpy_requires_explicit_band_names(self):
        anchor = make_anchor(width=4, height=4)
        with pytest.raises(ValueError, match="require explicit bands"):
            anchor.to_raster(np.ones((1, 4, 4), dtype="uint16"))

    def test_two_dimensional_raw_pixels_need_exactly_one_band_name(self):
        anchor = make_anchor(width=4, height=4)
        assert anchor.to_raster(np.ones((4, 4), dtype="uint16"), bands=["cls"]).bands == ("cls",)
        with pytest.raises(ValueError, match="exactly one band name"):
            anchor.to_raster(np.ones((4, 4), dtype="uint16"), bands=["a", "b"])

    def test_band_names_must_be_unique(self):
        anchor = make_anchor(width=4, height=4)
        with pytest.raises(ValueError, match="unique"):
            anchor.to_raster(np.ones((2, 4, 4), dtype="uint16"), bands=["B04", "B04"])

    def test_bands_are_rejected_when_a_dataarray_already_names_its_own(self, raster):
        with pytest.raises(ValueError, match="already names its own"):
            raster.anchor.to_raster(raster.data, bands=["B04"])

    def test_times_are_rejected_alongside_a_dataarray(self, raster):
        with pytest.raises(ValueError, match="belong to raw NumPy pixels"):
            raster.anchor.to_raster(raster.data, times=[datetime(2024, 1, 15)])

    def test_a_bandless_dataarray_takes_one_band_name(self, raster):
        """Derived layers arrive bandless — an index or mask off selected bands."""
        derived = raster.data.sel(band="B04") * 2
        named = raster.anchor.to_raster(derived, bands=["doubled"])
        assert named.bands == ("doubled",)
        assert named.data.dims == ("band", "y", "x")

    def test_naming_a_bandless_dataarray_keeps_it_lazy(self):
        lazy = make_lazy_raster(bands=("B04",))
        named = lazy.anchor.to_raster(lazy.data.sel(band="B04") * 2, bands=["doubled"])
        assert is_lazy(named)

    def test_a_bandless_dataarray_takes_exactly_one_name(self, raster):
        with pytest.raises(ValueError, match="exactly one band"):
            raster.anchor.to_raster(raster.data.sel(band="B04"), bands=["a", "b"])

    def test_geobox_must_match_the_anchor(self, raster):
        other = make_anchor(width=32, height=32)
        with pytest.raises(ValueError, match="doesn't match this anchor"):
            other.to_raster(raster.data)

    def test_time_coordinate_must_be_strictly_increasing(self):
        anchor = make_anchor(width=4, height=4, time=("2024-01-01", "2024-02-01"))
        out_of_order = [datetime(2024, 2, 1), datetime(2024, 1, 1)]
        with pytest.raises(ValueError, match="strictly increasing"):
            anchor.to_raster(
                np.ones((2, 1, 4, 4), dtype="float32"), bands=["B04"], times=out_of_order
            )

    def test_declared_span_must_cover_observed_labels(self):
        anchor = make_anchor(width=4, height=4, time="2024-01-15")
        with pytest.raises(ValueError, match="doesn't cover this data's own"):
            anchor.to_raster(
                np.ones((2, 1, 4, 4), dtype="float32"),
                bands=["B04"],
                times=[datetime(2024, 1, 15), datetime(2024, 3, 1)],
            )

    def test_undated_anchor_takes_its_span_from_the_labels(self):
        anchor = make_anchor(width=4, height=4, time=None)
        raster = anchor.to_raster(
            np.ones((2, 1, 4, 4), dtype="float32"),
            bands=["B04"],
            times=[datetime(2024, 1, 15), datetime(2024, 3, 1)],
        )
        assert raster.timespan is not None
        assert raster.timespan[0].date() == datetime(2024, 1, 15).date()
        assert raster.timespan[1].date() == datetime(2024, 3, 1).date()


class TestAnchorRequiresCrs:
    def test_crs_less_geobox_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="needs a CRS"):
            GeoAnchor(geobox=GeoBox((4, 4), Affine(10, 0, 0, 0, -10, 40), None))

    def test_crs_is_a_plain_string(self, anchor):
        assert anchor.crs == UTM
        assert isinstance(anchor.crs, str)


class TestHeaderOwnership:
    def test_a_stale_namespace_attr_cannot_shadow_the_anchors_own(self, raster):
        """`.attrs` mirrors the header each construction, so incoming namespace keys never win."""
        anchored = raster.rebase(tags={"real": "yes"})
        stamped = anchored.data.assign_attrs({"tags": {"stale": "yes"}})
        rebuilt = anchored.anchor.to_raster(stamped)
        assert rebuilt.tags == {"real": "yes"}
        assert rebuilt.data.attrs["tags"] == {"real": "yes"}

    def test_foreign_attrs_are_left_alone(self, raster):
        rebuilt = raster.anchor.to_raster(raster.data.assign_attrs({"mine": "keep"}))
        assert rebuilt.data.attrs["mine"] == "keep"

    def test_render_referencing_a_missing_band_is_rejected(self):
        raster = make_raster(bands=("B04",))
        with pytest.raises(ValueError, match="render.rgb_bands references missing bands"):
            raster.rebase(render={"rgb_bands": ("B04", "B08", "B04")})


class TestGridPreservingTransforms:
    """`astype`/`remap`/`compute` must leave the grid, and so the tiling stamp, intact."""

    @pytest.fixture
    def stamped(self) -> GeoTile:
        return next(iter(make_raster().tiles(tile_size_px=32)))

    def test_astype_keeps_the_tiling_stamp(self, stamped):
        assert stamped.astype("int32", nodata=-1).tiling is not None

    def test_remap_keeps_the_tiling_stamp(self, stamped):
        assert stamped.remap({1: 2}).tiling is not None

    def test_compute_keeps_the_tiling_stamp(self, stamped):
        assert stamped.compute().tiling is not None

    def test_tile_has_no_grid_changing_transforms(self):
        assert not hasattr(GeoTile, "reproject")
        assert not hasattr(GeoTile, "reproject_like")
        assert not hasattr(GeoTile, "crop")

    def test_raster_owns_them_instead(self):
        assert hasattr(GeoRaster, "reproject")
        assert hasattr(GeoRaster, "reproject_like")
        assert hasattr(GeoRaster, "crop")


class TestDtypeAndNodata:
    def test_astype_preserves_a_compatible_sentinel(self):
        cast = make_raster(nodata=0).astype("float32")
        assert cast.dtype == np.dtype("float32")
        assert cast.nodata == 0.0

    def test_astype_rewrites_pixels_when_the_sentinel_changes(self):
        raster = make_raster(bands=("cls",), dtype="uint8", nodata=0, width=4, height=4)
        cast = raster.astype("int32", nodata=-1)
        assert cast.nodata == -1
        assert (cast.data.values == -1).sum() == (raster.data.values == 0).sum()

    def test_nan_nodata_cannot_be_cast_to_an_integer_without_a_new_sentinel(self):
        raster = make_raster(dtype="float32", nodata=np.nan, width=4, height=4)
        with pytest.raises(ValueError, match="needs an explicit integer nodata"):
            raster.astype("int32", nodata=None)

    def test_remap_keeps_dtype_and_nodata(self):
        anchor = make_anchor(width=4, height=4)
        pixels = np.tile(np.array([[1, 2], [2, 3]], dtype="uint8"), (2, 2))[None, ...]
        raster = anchor.to_raster(pixels, bands=["cls"]).rebase(nodata=255)
        remapped = raster.remap({1: 7, 2: 8})
        assert remapped.dtype == np.dtype("uint8")
        assert remapped.nodata == 255
        assert set(np.unique(remapped.data.values)) == {7, 8, 3}

    def test_remap_reads_off_the_original_so_order_cannot_chain(self):
        anchor = make_anchor(width=2, height=2)
        raster = anchor.to_raster(np.array([[1, 2], [1, 2]], dtype="uint8"), bands=["cls"]).rebase(nodata=255)
        assert set(np.unique(raster.remap({1: 2, 2: 3}).data.values)) == {2, 3}

    def test_remap_refuses_to_consume_the_declared_nodata(self):
        raster = make_raster(bands=("cls",), dtype="uint8", nodata=255, width=4, height=4)
        with pytest.raises(ValueError, match="cannot use declared nodata"):
            raster.remap({255: 0})

    def test_remap_rejects_a_non_integer_raster(self):
        with pytest.raises(TypeError, match="needs an integer raster"):
            make_raster(dtype="float32", nodata=0).remap({1: 2})

    def test_nodata_must_be_representable_by_the_dtype(self):
        anchor = make_anchor(width=4, height=4)
        raster = anchor.to_raster(np.ones((1, 4, 4), dtype="uint8"), bands=["cls"])
        with pytest.raises(ValueError, match="outside"):
            raster.rebase(nodata=99999)


class TestBandNaming:
    def test_rename_bands_keeps_order_and_pixels(self):
        renamed = make_raster().rename_bands({"B04": "red"})
        assert renamed.bands == ("red", "B08")

    def test_rename_bands_follows_render_references(self):
        raster = make_raster().rebase(render={"rgb_bands": ("B04", "B08", "B04")})
        renamed = raster.rename_bands({"B04": "red"})
        assert renamed.render is not None
        assert renamed.render.rgb_bands == ("red", "B08", "red")

    def test_rename_bands_rejects_an_unknown_source(self):
        with pytest.raises(KeyError):
            make_raster().rename_bands({"nope": "red"})

    def test_rename_bands_rejects_a_duplicate_result(self):
        with pytest.raises(ValueError, match="unique"):
            make_raster().rename_bands({"B04": "B08"})

    def test_rename_bands_rejects_an_empty_replacement(self):
        with pytest.raises(ValueError, match="must not be empty"):
            make_raster().rename_bands({"B04": "  "})


class TestGeoboxTolerance:
    def test_float_noise_below_a_pixel_still_matches(self):
        raster = make_raster(width=8, height=8)
        drifted = raster.data.assign_coords(
            x=raster.data.x.values + 1e-9, y=raster.data.y.values + 1e-9
        )
        # reconstructing the anchor from noisy coords must not read as a different grid
        assert GeoRaster(data=drifted, anchor=raster.anchor).shape == (8, 8)

    def test_a_real_half_pixel_offset_does_not(self):
        raster = make_raster(width=8, height=8)
        shifted = raster.data.assign_coords(x=raster.data.x.values + 5.0)
        with pytest.raises(ValueError, match="isn't this data's own"):
            GeoRaster(data=shifted, anchor=raster.anchor)


def test_geobox_helper_places_the_origin_exactly():
    """The builders' own contract — other tests assert against these coordinates."""
    box = make_geobox(width=8, height=8, resolution=10)
    assert box.boundingbox.bbox == (500_000.0, 5_000_000.0, 500_080.0, 5_000_080.0)
