from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.raster import raster as build_raster
from geosave_engine.geodata.core.stack import stack as build_stack
from geosave_engine.geodata.transform.warp import reproject, reproject_match

UTM = "EPSG:32633"
WGS84 = "EPSG:4326"


def utm_box(size: int = 256, resolution: int = 10) -> GeoBox:
    """Build a projected grid of `size` square pixels."""
    span = size * resolution
    return GeoBox.from_bbox(
        (300000, 5000000, 300000 + span, 5000000 + span), crs=UTM, resolution=resolution
    )


def scene() -> xr.Dataset:
    """Build a two-band uint16 raster on the projected grid."""
    box = utm_box()
    return build_raster(
        {"red": np.ones(box.shape, "uint16"), "nir": np.ones(box.shape, "uint16")},
        box,
        nodata=0,
    )


def landcover() -> xr.Dataset:
    """Build a raster whose values are class codes."""
    box = utm_box()
    coded = build_raster({"landcover": np.ones(box.shape, "uint8")}, box)
    return attrs.rebase(
        coded, attrs.Legend(class_map={1: "water", 2: "urban"}), target="landcover"
    )


def test_a_geobox_target_lands_exactly_on_it() -> None:
    target = utm_box().to_crs(WGS84)

    warped = reproject_match(scene(), target)

    assert warped.gs.geobox == target


def test_a_warped_raster_reports_the_crs_it_landed_on() -> None:
    source = build_raster({"elevation": np.ones(utm_box().shape, "float32")}, utm_box())

    warped = reproject(source, WGS84)

    assert warped.gs.crs.epsg == 4326
    # odc's own Dataset path leaves the source CRS in spatial_ref's attrs.
    assert source.odc.reproject(WGS84).odc.crs.epsg == 32633


def test_a_warped_raster_stacks_with_the_grid_it_targeted() -> None:
    imagery = scene()
    dem_box = GeoBox.from_bbox((12.9, 45.0, 13.2, 45.2), crs=WGS84, resolution=0.0003)
    dem = build_raster({"elevation": np.ones(dem_box.shape, "float32")}, dem_box)

    warped = reproject_match(dem, imagery, resampling="bilinear")

    assert build_stack({"optical": imagery, "dem": warped}).gs.groups == (
        "optical",
        "dem",
    )


def test_warping_keeps_the_raster_and_variable_attrs() -> None:
    source = attrs.rebase(scene(), attrs.ACDD(title="Sentinel-2 Level-2A"))

    warped = reproject(source, WGS84)

    assert warped.gs.attrs.root.get(attrs.ACDD).title == "Sentinel-2 Level-2A"
    assert warped.red.attrs["_FillValue"] == 0


def test_warping_a_band_returns_a_band() -> None:
    warped = reproject(scene().red, WGS84)

    assert isinstance(warped, xr.DataArray)
    assert warped.gs.crs.epsg == 4326


def test_warping_a_stack_leaves_every_group_on_one_grid() -> None:
    source = build_stack({"optical": scene(), "cover": landcover()})

    warped = reproject(source, WGS84)

    grids = {name: group.gs.geobox for name, group in warped.gs.rasters.items()}
    assert warped.gs.groups == ("optical", "cover")
    assert len(set(grids.values())) == 1
    assert warped.gs.crs.epsg == 4326


def test_warping_a_stack_onto_a_geobox_lands_every_group_on_it() -> None:
    target = utm_box().to_crs(WGS84)
    source = build_stack({"optical": scene(), "cover": landcover()})

    warped = reproject_match(source, target)

    assert all(group.gs.geobox == target for group in warped.gs.rasters.values())


@pytest.mark.parametrize("resampling", ["bilinear", "cubic", "average"])
def test_a_blending_kernel_refuses_class_codes(resampling: str) -> None:
    with pytest.raises(ValueError, match="carry a class map"):
        reproject(landcover(), WGS84, resampling=resampling)


@pytest.mark.parametrize("resampling", ["nearest", "mode"])
def test_a_value_preserving_kernel_accepts_class_codes(resampling: str) -> None:
    warped = reproject(landcover(), WGS84, resampling=resampling)

    assert set(np.unique(warped.landcover.values)) <= {0, 1, 2}


def test_a_blending_kernel_names_the_group_holding_class_codes() -> None:
    source = build_stack({"optical": scene(), "cover": landcover()})

    with pytest.raises(ValueError, match=r"cover/landcover"):
        reproject(source, WGS84, resampling="bilinear")


def test_reproject_honours_a_requested_resolution() -> None:
    warped = reproject(scene(), WGS84, resolution=0.001)

    assert warped.gs.resolution.x == 0.001


def test_matching_adopts_a_rasters_grid_without_naming_a_geobox() -> None:
    imagery = scene()
    dem_box = GeoBox.from_bbox((12.9, 45.0, 13.2, 45.2), crs=WGS84, resolution=0.0003)
    dem = build_raster({"elevation": np.ones(dem_box.shape, "float32")}, dem_box)

    assert reproject_match(dem, imagery).gs.geobox == imagery.gs.geobox


def test_a_variable_off_the_grid_is_refused_by_name() -> None:
    source = scene()
    source["profile"] = ("depth", np.arange(3.0))

    with pytest.raises(ValueError, match="'profile' spans"):
        reproject(source, WGS84)


def test_warping_refuses_a_raster_carrying_no_grid() -> None:
    unreferenced = xr.Dataset({"band": (("y", "x"), np.zeros((4, 4), "uint8"))})

    with pytest.raises(ValueError, match="georegistered|locatable grid"):
        reproject(unreferenced, WGS84)
