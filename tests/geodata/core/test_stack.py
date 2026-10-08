from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.stac import create_stack_items
from geosave_engine.geodata import read_raster, read_stack
from geosave_engine.geodata.core.stack import map_groups, stack as build_stack
from geosave_engine.geodata.io import gdal

from tests.geodata.conftest import build_raster


@pytest.mark.parametrize("times", [0, 2])
@pytest.mark.parametrize("split_bands", [False, True])
def test_cog_export_retains_groups_with_repeated_band_names(
    tmp_path: Path, times: int, split_bands: bool
) -> None:
    image = build_raster(times=times)
    label = (image // 1000).astype("uint8")
    written = build_stack({"image.v2": image, "label.v1": label})
    root = tmp_path / "scene"
    root.mkdir()
    unrelated = root / "unrelated.tif"
    unrelated.touch()

    saved = written.gs.to_cog(root, split_bands=split_bands)

    assert isinstance(saved, dict)
    assert list(saved) == ["image.v2", "label.v1"]
    for group, paths in saved.items():
        assert len(paths) == (times or 1) * (2 if split_bands else 1)
        assert unrelated not in paths
        source = written.gs.rasters[group]
        for path in paths:
            assert group in str(path.relative_to(root))
            with gdal.read(path) as restored:
                expected = source.sel(time=restored.time) if times else source
                assert restored.gs.geobox == source.gs.geobox
                for name in restored.data_vars:
                    assert restored[name].dtype == source[name].dtype
                    np.testing.assert_array_equal(restored[name], expected[name])


def test_timeless_cog_groups_with_file_suffixes_do_not_collide(tmp_path: Path) -> None:
    image = build_raster()
    label = (image // 1000).astype("uint8")
    written = build_stack({"image": image, "image.tif": label, "image.TIFF": image})

    saved = written.gs.to_cog(tmp_path)

    assert saved == {
        "image": (tmp_path / "image.tif",),
        "image.tif": (tmp_path / "image.tif.tif",),
        "image.TIFF": (tmp_path / "image.TIFF.tif",),
    }
    with gdal.read(saved["image.tif"][0]) as restored:
        np.testing.assert_array_equal(restored.red, label.red)


def test_a_stack_reads_back_from_what_its_cog_export_returns(tmp_path: Path) -> None:
    image = build_raster(times=2)
    written = build_stack({"image": image, "label": (image // 1000).astype("uint8")})

    restored = read_stack(written.gs.to_cog(tmp_path / "scene"))

    assert restored.gs.groups == ("image", "label")
    np.testing.assert_array_equal(
        restored.gs.rasters["label"].red, written.gs.rasters["label"].red
    )


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
    expected = ("y", "x", "spatial_ref")
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


def test_a_stack_without_a_shared_grid_names_no_anchor() -> None:
    loose = build_stack(
        {"whole": build_raster(), "geographic": build_raster(crs="EPSG:4326")}
    )

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
    paths = stack.gs.to_zarr(tmp_path / "stack")

    restored = read_stack(paths)

    assert list(paths) == list(stack.gs.groups)
    assert paths["optical"] == tmp_path / "stack/optical.zarr"
    assert restored.gs.groups == stack.gs.groups
    assert restored.gs.geobox == stack.gs.geobox
    assert restored["optical"].dataset.red.dtype == np.dtype("uint16")


def test_every_group_opens_on_its_own(tmp_path: Path, stack: xr.DataTree) -> None:
    paths = stack.gs.to_netcdf(tmp_path / "solo")

    store = paths["optical"]
    solo = read_raster(store)

    assert store == tmp_path / "solo/optical.nc"
    assert solo.gs.geobox == stack.gs.geobox
    assert "spatial_ref" in solo.coords


def test_a_saved_group_is_not_replaced_unasked(tmp_path, stack: xr.DataTree) -> None:
    stack.gs.to_zarr(tmp_path / "stack")

    with pytest.raises(FileExistsError):
        stack.gs.to_zarr(tmp_path / "stack")

    assert list(stack.gs.to_zarr(tmp_path / "stack", overwrite=True)) == list(
        stack.gs.groups
    )


def test_a_deferred_stack_write_resolves_to_its_stores(
    tmp_path, stack: xr.DataTree
) -> None:
    deferred = stack.chunk().gs.to_zarr(tmp_path / "later", compute=False)

    paths = deferred.compute()

    assert list(paths) == list(stack.gs.groups)
    np.testing.assert_array_equal(
        read_stack(paths)["optical"].red, stack["optical"].red
    )


def test_a_single_store_is_not_a_stack(tmp_path, stack: xr.DataTree) -> None:
    store = stack.gs.to_zarr(tmp_path / "stack")["optical"]

    with pytest.raises(ValueError, match="read_raster"):
        read_stack(store)


def test_a_stack_spans_from_its_earliest_group_to_its_latest() -> None:
    dated = build_raster(times=2)
    timeless = build_raster()

    assert build_stack({"dem": timeless}).gs.timespan is None
    assert (
        build_stack({"optical": dated, "dem": timeless}).gs.timespan
        == dated.gs.timespan
    )


def test_a_stack_lists_every_instant_any_group_reaches() -> None:
    early = build_raster(times=2)
    late = early.assign_coords(time=early.time + np.timedelta64(1, "D"))

    mixed = build_stack({"optical": early, "dem": build_raster()}).gs.times
    joint = build_stack({"a": early, "b": late}).gs.times

    assert build_stack({"dem": build_raster()}).gs.times is None
    assert mixed is not None and mixed.equals(early.gs.times)
    assert joint is not None
    assert joint.strftime("%Y%m%d").tolist() == ["20250601", "20250602", "20250603"]


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


def test_a_saved_stack_indexes_one_item_per_group_scene(tmp_path: Path) -> None:
    optical = build_raster(times=2)
    sample = build_stack({"optical": optical, "dem": build_raster()})

    items = create_stack_items(sample.gs.to_cog(tmp_path / "s0"), name="s0")

    assert [item.id for item in items] == [
        "s0/optical_20250601T000000",
        "s0/optical_20250602T000000",
        "s0/dem",
    ]
    assert [item.collection_id for item in items] == ["optical", "optical", "dem"]
    assert {item.properties["geosave:stack"] for item in items} == {"s0"}
    assert items[2].assets["image"].href == str(tmp_path / "s0/dem.tif")


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_a_stack_store_restores_each_group_through_the_table(tmp_path, suffix) -> None:
    from geosave_engine.geodata.stac import table

    optical = build_raster(times=2)
    sample = build_stack({"optical": optical, "label": build_raster(times=2)})
    write = sample.gs.to_zarr if suffix == ".zarr" else sample.gs.to_netcdf

    items = create_stack_items(write(tmp_path / "s0"), name="s0")

    rows = table.from_items(items)
    restored = build_stack(
        {
            group: table.load(table.to_items(rows[rows["collection"] == group]))
            for group in sample.gs.groups
        }
    )
    assert restored.gs.groups == sample.gs.groups
    np.testing.assert_array_equal(restored["optical"].red, optical.red)
    restored.close()


def test_shared_geographic_stack_spans_y_x() -> None:
    from affine import Affine
    from odc.geo import GeoBox
    from odc.geo.xr import xr_coords

    dims = ("y", "x")
    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    scene = xr.Dataset(
        {"red": (dims, np.ones((2, 3)))}, coords=xr_coords(grid, dims=dims)
    )
    scene.red.encoding["grid_mapping"] = "spatial_ref"

    tree = build_stack({"optical": scene, "dem": scene.rename({"red": "elevation"})})

    assert tree.gs.geobox == grid
    assert tree["optical"].dataset.red.dims == dims
    assert tree["dem"].dataset.elevation.dims == dims
