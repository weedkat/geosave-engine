from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.stack import map_groups, stack as build_stack
from geosave_engine.geodata.io import zarr

from tests.geodata.conftest import build_raster


def test_group_extraction_preserves_all_root_coordinates():
    raster = build_raster(times=2).isel(time=0)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords=raster.coords),
            "/image": raster.drop_vars(list(raster.coords)),
        }
    )
    xr.testing.assert_equal(tree.gs.rasters["image"], raster)
    mapped = map_groups(tree, lambda ds: ds.copy())
    xr.testing.assert_equal(mapped.gs.rasters["image"], raster)


def test_stack_carries_the_shared_grid_at_its_root() -> None:
    raster = build_raster()

    built = build_stack({"optical": raster[["red"]], "dem": raster[["nir"]]})

    assert built.gs.groups == ("optical", "dem")
    assert built.gs.geobox == raster.gs.geobox
    expected = (*raster.gs.geobox.dimensions, "spatial_ref")
    assert set(expected) <= set(built.dataset.coords)
    assert not built.dataset.data_vars


def test_stack_names_are_not_restricted_to_identifiers() -> None:
    raster = build_raster()

    built = build_stack({"sentinel-2-l2a": raster[["red"]], "ndvi.v2": raster[["nir"]]})

    assert built.gs.groups == ("sentinel-2-l2a", "ndvi.v2")


def test_stack_refuses_an_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_stack({})


def test_stack_refuses_a_name_spelling_a_path() -> None:
    raster = build_raster()

    with pytest.raises(ValueError, match="spell paths rather than group names"):
        build_stack(
            {"sentinel-2/r10": raster[["red"]], "sentinel-2/r20": raster[["nir"]]}
        )


def test_groups_covering_different_ground_leave_the_root_bare() -> None:
    whole = build_raster()
    corner = build_raster().isel(y=slice(0, 1), x=slice(0, 1))

    loose = build_stack({"whole": whole, "corner": corner})

    assert loose.gs.groups == ("whole", "corner")
    assert loose.gs.geobox is None
    assert loose.gs.rasters["corner"].gs.geobox == corner.gs.geobox


def test_groups_in_different_crs_leave_the_root_bare() -> None:
    projected = build_raster(crs="EPSG:32749")
    geographic = build_raster(crs="EPSG:4326")

    loose = build_stack({"projected": projected, "geographic": geographic})

    assert loose.gs.geobox is None
    assert loose.gs.rasters["geographic"].gs.crs.epsg == 4326


def test_an_unplaced_raster_leaves_the_root_bare() -> None:
    unplaced = xr.Dataset({"red": (("y", "x"), np.zeros((2, 2), "uint16"))})

    loose = build_stack({"unplaced": unplaced})

    assert loose.gs.groups == ("unplaced",)
    assert loose.gs.geobox is None


def test_a_stack_without_a_shared_grid_names_no_axes_and_no_anchor() -> None:
    loose = build_stack(
        {"whole": build_raster(), "geographic": build_raster(crs="EPSG:4326")}
    )

    with pytest.raises(ValueError, match="publishes no grid at its root"):
        _ = loose.gs.grid_dims
    with pytest.raises(ValueError, match="publishes no grid at its root"):
        _ = loose.gs.anchor


def test_xarray_refuses_a_misaligned_group_after_construction(
    stack: xr.DataTree,
) -> None:
    shifted = build_raster()
    shifted = shifted.assign_coords(y=shifted.y.values + 12_345.0)

    with pytest.raises(ValueError, match="not aligned"):
        stack["shifted"] = xr.DataTree(shifted)


def test_stack_round_trip_preserves_groups_and_grid(
    tmp_path: Path, stack: xr.DataTree
) -> None:
    destination = zarr.write(stack, tmp_path / "stack.zarr")

    restored = zarr.read_stack(destination)

    assert set(restored.gs.groups) == {"optical", "infrared"}
    assert restored.gs.geobox == stack.gs.geobox
    assert restored["optical"].dataset.red.dtype == np.dtype("uint16")


def test_every_group_opens_on_its_own(tmp_path: Path, stack: xr.DataTree) -> None:
    destination = zarr.write(stack, tmp_path / "solo.zarr")

    solo = zarr.read(destination, group="optical")

    assert solo.gs.geobox == stack.gs.geobox
    assert "spatial_ref" in solo.coords


def test_a_stack_spans_from_its_earliest_group_to_its_latest() -> None:
    dated = build_raster(times=2)
    timeless = build_raster()

    assert build_stack({"dem": timeless}).gs.timespan is None
    assert (
        build_stack({"optical": dated, "dem": timeless}).gs.timespan
        == dated.gs.timespan
    )


def test_a_stack_anchors_on_its_shared_grid_and_joint_span() -> None:
    dated = build_raster(times=2)
    scene = build_stack({"optical": dated, "dem": build_raster()})

    assert scene.gs.anchor.geobox == scene.gs.geobox
    assert scene.gs.anchor.timespan == dated.gs.timespan
    assert build_stack({"dem": build_raster()}).gs.anchor.timespan is None


def test_map_groups_changes_every_group_and_keeps_the_root_attrs() -> None:
    raster = build_raster()
    built = build_stack({"optical": raster[["red"]], "dem": raster[["nir"]]})
    titled = built.gs.rebase(attrs.ACDD(title="scene"))

    halved = map_groups(titled, lambda group: group.isel(x=slice(0, 1)))

    assert halved.gs.groups == ("optical", "dem")
    assert halved.gs.geobox.shape.x == 1
    assert halved.gs.attrs.root.get(attrs.ACDD).title == "scene"


def test_map_groups_rebuilds_a_stack_that_shares_no_grid() -> None:
    raster = build_raster()
    coarse = raster[["nir"]].isel(x=slice(0, 1))
    built = build_stack({"optical": raster[["red"]], "dem": coarse})
    titled = built.gs.rebase(attrs.ACDD(title="scene"))

    renamed = map_groups(
        titled,
        lambda group: group.rename({name: f"{name}_raw" for name in group.data_vars}),
    )

    assert renamed.gs.geobox is None
    assert renamed.gs.variables == ("optical/red_raw", "dem/nir_raw")
    assert renamed.gs.attrs.root.get(attrs.ACDD).title == "scene"


def test_a_stack_reads_the_pixel_operations_through_gs(stack: xr.DataTree) -> None:
    import geopandas as gpd
    import shapely

    valid = np.array([[True, False], [True, True]])
    left, bottom, right, top = stack.gs.bounds.bbox
    column = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs=stack.gs.crs,
    )

    assert stack.gs.unpack().gs.groups == stack.gs.groups
    assert stack.gs.to_nan().gs.groups == stack.gs.groups
    assert stack.gs.mask(valid, fill=0).gs.rasters["optical"].red.values[0, 1] == 0
    assert stack.gs.crop(column, mask=False).gs.geobox.shape == (2, 1)


def test_a_stack_reprojects_onto_a_target_rasters_own_grid(
    stack: xr.DataTree,
) -> None:
    target = build_raster(crs="EPSG:32748")

    warped = stack.gs.reproject(target)

    assert warped.gs.groups == stack.gs.groups
    assert warped.gs.geobox == target.gs.geobox
    assert stack.gs.reproject("EPSG:3857").gs.crs.epsg == 3857
