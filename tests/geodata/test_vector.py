from datetime import UTC
from pathlib import Path
import shutil
from typing import cast

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from pandas import isna
from shapely.geometry import Point, box

from geosave_engine.geodata import GeoAnchor, GeoVector, read_vector
from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.core.vector import SpatialPredicate, XarrayField
from geosave_engine.geodata.transform.vector import rasterize, vectorize

from .conftest import build_raster


def test_empty_vector_has_a_crs_and_no_footprint() -> None:
    vector = GeoVector.empty("EPSG:4326")

    assert len(vector) == 0
    assert vector.crs.to_epsg() == 4326
    with pytest.raises(ValueError, match="empty"):
        _ = vector.footprint


def test_geometry_properties_become_columns() -> None:
    vector = GeoVector.from_geometry(
        Point(112.15, -8.05), place_name="Malang", split="train"
    )

    assert vector.gdf.loc[0, "place_name"] == "Malang"
    assert vector.gdf.loc[0, "split"] == "train"


def test_anchor_record_has_exact_grid_and_time() -> None:
    anchor = GeoAnchor.from_coordinates(
        -8.05, 112.15, shape=(3, 4), resolution=10, timespan="2025-01"
    )

    vector = GeoVector.from_anchor(anchor, crs="EPSG:4326")
    row = vector.gdf.iloc[0]

    assert row.grid_crs == str(anchor.crs)
    assert tuple(row.grid_transform) == tuple(anchor.geobox.transform)[:6]
    assert (row.grid_height, row.grid_width) == (3, 4)
    assert row.start_datetime.tzinfo is UTC
    assert row.end_datetime.tzinfo is UTC
    assert vector.crs.to_epsg() == 4326


def test_timeless_anchor_uses_null_time_columns() -> None:
    anchor = GeoAnchor.from_coordinates(-8.05, 112.15, shape=2, resolution=10)

    row = GeoVector.from_anchor(anchor).gdf.iloc[0]

    assert isna(row.start_datetime)
    assert isna(row.end_datetime)


def test_xarray_record_keeps_semantic_geometry_and_native_grid() -> None:
    raster = build_raster(times=2)
    semantic = box(112.0, -8.2, 112.2, -8.0)

    vector = GeoVector.from_xarray(
        raster,
        geometry=semantic,
        crs="EPSG:4326",
        path="rasters/prediction.zarr",
        fields=("time", "grid", "variables"),
        model="forest-v1",
    )
    row = vector.gdf.iloc[0]

    assert row.geometry.equals(semantic)
    assert row.grid_crs == str(raster.gs.anchor.crs)
    assert tuple(row.variables) == ("red", "nir")
    assert row.path == "rasters/prediction.zarr"
    assert row.model == "forest-v1"


def test_xarray_registration_does_not_compute_pixels() -> None:
    raster = build_raster().chunk({"y": 1, "x": 1})
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        vector = GeoVector.from_xarray(raster)

    assert len(vector) == 1
    assert started == []


def test_xarray_fields_are_explicit() -> None:
    vector = GeoVector.from_xarray(build_raster(), fields=())

    assert set(vector.gdf) == {"geometry"}
    with pytest.raises(ValueError, match="unsupported vector fields"):
        GeoVector.from_xarray(
            build_raster(), fields=cast("tuple[XarrayField, ...]", ("attrs",))
        )


def test_datatree_registration_keeps_grouped_variable_identity(stack) -> None:
    vector = GeoVector.from_xarray(stack, fields=("variables",))

    assert tuple(vector.gdf.loc[0, "variables"]) == (
        "optical/red",
        "infrared/nir",
    )


def test_derived_properties_cannot_be_replaced() -> None:
    anchor = GeoAnchor.from_coordinates(
        -8.05, 112.15, shape=2, resolution=10
    )

    with pytest.raises(ValueError, match="geometry"):
        GeoVector.from_anchor(anchor, geometry="caller-value")
    with pytest.raises(ValueError, match="grid_crs"):
        GeoVector.from_anchor(
            anchor,
            grid_crs="caller-value",
        )


def test_concat_unions_columns_resets_index_and_keeps_duplicates() -> None:
    first = GeoVector.from_geometry(Point(0, 0), name="first")
    renamed = GeoVector(
        GeoVector.from_geometry(Point(1, 1), score=4).gdf.rename_geometry(
            "footprint"
        )
    )

    combined = GeoVector.concat([first, renamed, first])

    assert list(combined.gdf.index) == [0, 1, 2]
    assert list(combined.gdf.geometry) == [Point(0, 0), Point(1, 1), Point(0, 0)]
    assert list(combined.gdf.name.iloc[[0, 2]]) == ["first", "first"]
    assert combined.gdf.loc[1, "score"] == 4


def test_concat_requires_one_explicit_crs_plane() -> None:
    geographic = GeoVector.from_geometry(Point(0, 0))
    projected = GeoVector.from_geometry(Point(0, 0), crs="EPSG:3857")

    with pytest.raises(ValueError, match="to_crs"):
        GeoVector.concat([geographic, projected])


def test_upsert_replaces_caller_key_and_appends_new_key() -> None:
    old = GeoVector.from_geometry(Point(0, 0), record_id="a", status="old")
    untouched = GeoVector.from_geometry(Point(1, 1), record_id="b", status="same")
    replacement = GeoVector.from_geometry(Point(2, 2), record_id="a", status="new")
    added = GeoVector.from_geometry(Point(3, 3), record_id="c", status="new")
    catalog = GeoVector.concat([old, untouched])

    result = catalog.upsert(GeoVector.concat([replacement, added]), on="record_id")

    assert list(result.gdf.record_id) == ["b", "a", "c"]
    assert list(result.gdf.status) == ["same", "new", "new"]


@pytest.mark.parametrize("problem", ["missing", "null", "duplicate"])
def test_upsert_rejects_invalid_incoming_identity(problem: str) -> None:
    existing = GeoVector.from_geometry(Point(0, 0), record_id="a")
    if problem == "missing":
        incoming = GeoVector.from_geometry(Point(1, 1))
    elif problem == "null":
        incoming = GeoVector.from_geometry(Point(1, 1), record_id=None)
    else:
        incoming = GeoVector.concat(
            [
                GeoVector.from_geometry(Point(1, 1), record_id="b"),
                GeoVector.from_geometry(Point(2, 2), record_id="b"),
            ]
        )

    with pytest.raises((KeyError, ValueError)):
        existing.upsert(incoming, on="record_id")


def test_query_returns_original_intersecting_rows_in_source_order() -> None:
    collection = GeoVector(
        gpd.GeoDataFrame(
            {"name": ["left", "middle", "right"]},
            geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2), box(8, 0, 10, 2)],
            crs="EPSG:4326",
        )
    )

    result = collection.query(box(1, -1, 3, 3), predicate="intersects")

    assert list(result.gdf.name) == ["left", "middle"]
    assert result.gdf.iloc[0].geometry.equals(collection.gdf.iloc[0].geometry)


def test_query_predicate_describes_each_row_relative_to_target() -> None:
    collection = GeoVector(
        gpd.GeoDataFrame(
            {"name": ["inside", "outside", "container"]},
            geometry=[box(1, 1, 2, 2), box(5, 5, 6, 6), box(-1, -1, 4, 4)],
            crs="EPSG:4326",
        )
    )
    target = box(0, 0, 3, 3)

    assert list(collection.query(target, predicate="within").gdf.name) == [
        "inside"
    ]
    assert list(collection.query(target, predicate="contains").gdf.name) == [
        "container"
    ]


def test_query_transforms_only_an_anchor_footprint() -> None:
    collection = GeoVector.from_geometry(
        box(112.0, -8.2, 112.2, -8.0), crs="EPSG:4326", name="place"
    )
    anchor = GeoAnchor.from_coordinates(-8.1, 112.1, shape=4, resolution=100)

    result = collection.query(anchor)

    assert list(result.gdf.name) == ["place"]
    assert result.crs.to_epsg() == 4326


def test_query_empty_collection_preserves_schema() -> None:
    empty = GeoVector.empty("EPSG:4326")

    result = empty.query(box(0, 0, 1, 1))

    assert len(result) == 0
    assert result.crs.to_epsg() == 4326


def test_query_rejects_unknown_predicate() -> None:
    vector = GeoVector.from_geometry(Point(0, 0))

    with pytest.raises(ValueError, match="predicate"):
        vector.query(
            Point(0, 0), predicate=cast("SpatialPredicate", "touch-ish")
        )


def test_asset_path_round_trips_relative_to_movable_catalog(tmp_path: Path) -> None:
    original = tmp_path / "original"
    (original / "rasters" / "prediction.zarr").mkdir(parents=True)
    catalog = GeoVector.from_geometry(
        Point(0, 0), path=original / "rasters" / "prediction.zarr"
    )

    catalog.to_geoparquet(original / "catalog.parquet")
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    restored = read_vector(moved / "catalog.parquet")

    assert restored.gdf.iloc[0].path == moved / "rasters" / "prediction.zarr"
    assert not hasattr(restored, "source")


def test_absolute_asset_outside_catalog_remains_absolute(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "shared" / "prediction.zarr"
    catalog = GeoVector.from_geometry(Point(0, 0), path=outside)

    written = catalog.to_geoparquet(root / "catalog.parquet")
    restored = read_vector(written)

    assert restored.gdf.iloc[0].path == outside


def test_relative_parent_asset_expands_without_dereferencing(tmp_path: Path) -> None:
    root = tmp_path / "dataset" / "catalogs"
    root.mkdir(parents=True)
    catalog = GeoVector.from_geometry(
        Point(0, 0), path="../shared/missing.zarr"
    )

    written = catalog.to_geoparquet(root / "catalog.parquet")
    restored = read_vector(written)

    assert (
        restored.gdf.iloc[0].path
        == tmp_path / "dataset" / "shared" / "missing.zarr"
    )


def test_unsaved_vector_keeps_the_caller_pointer() -> None:
    vector = GeoVector.from_geometry(Point(0, 0), path="assets/a.zarr")

    assert vector.gdf.iloc[0].path == "assets/a.zarr"
    assert not hasattr(vector, "resolve_path")


def test_vectorize_creates_one_row_per_connected_flag_region(raster) -> None:
    flags = array(
        np.array([[1, 0, 1], [1, 0, 0], [2, 2, 0]], dtype="uint8"),
        GeoAnchor.from_bbox(
            raster.gs.geobox.boundingbox,
            shape=(3, 3),
            crs=raster.gs.crs,
        ).geobox,
        nodata=0,
    )

    vector = GeoVector.vectorize(flags, value_name="class_id")

    assert list(vector.gdf.class_id) == [1, 1, 2]
    assert vector.crs == flags.gs.crs
    assert vector.gdf.geometry.is_valid.all()


def test_vectorize_preserves_the_value_schema_when_every_pixel_is_nodata(
    raster,
) -> None:
    flags = array(
        np.zeros(raster.gs.geobox.shape, dtype="uint8"),
        raster.gs.geobox,
        nodata=0,
    )

    vector = GeoVector.vectorize(flags, value_name="class_id")
    restored = vector.rasterize(
        like=flags,
        column="class_id",
        fill=0,
        dtype="uint8",
    )

    assert list(vector.gdf) == ["class_id", "geometry"]
    assert vector.gdf.class_id.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(restored.values, np.zeros(flags.shape, "uint8"))


def test_vector_method_delegates_to_vector_transform(raster) -> None:
    flags = raster.red.astype("uint8")

    direct = vectorize(flags)
    through_method = GeoVector.vectorize(flags)

    assert direct.gdf.equals(through_method.gdf)


def test_vectorize_computes_dask_flags_explicitly(raster) -> None:
    flags = raster.red.chunk({"y": 1, "x": 1})
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        GeoVector.vectorize(flags)

    assert started


def test_vectorize_requires_one_spatial_plane(raster) -> None:
    flags = raster.red.expand_dims(time=["2025-01-01"])

    with pytest.raises(ValueError, match="two-dimensional"):
        GeoVector.vectorize(flags)


def test_vectorize_rejects_unsupported_values(raster) -> None:
    flags = raster.red.astype("complex64")

    with pytest.raises(ValueError, match="dtype"):
        GeoVector.vectorize(flags)


def test_vectorize_rejects_same_shape_mask_on_another_grid(raster) -> None:
    shifted = raster.red.assign_coords(x=raster.x + 10)

    with pytest.raises(ValueError, match="exact flags grid"):
        GeoVector.vectorize(raster.red, mask=shifted)


def test_rasterize_values_back_onto_reference_grid(raster) -> None:
    flags = array(
        np.array([[1, 1], [2, 0]], dtype="uint8"),
        raster.gs.geobox,
        nodata=0,
    )
    vector = GeoVector.vectorize(flags, value_name="class_id")

    restored = vector.rasterize(
        like=raster,
        column="class_id",
        fill=0,
        dtype="uint8",
    )

    np.testing.assert_array_equal(restored.values, flags.values)
    assert restored.gs.geobox == raster.gs.geobox
    assert restored.name == "class_id"


def test_raster_method_delegates_to_vector_transform(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.anchor.geobox.extent)

    direct = rasterize(vector, raster)
    through_method = vector.rasterize(raster)

    xr.testing.assert_identical(direct, through_method)


def test_rasterize_without_column_returns_boolean_presence(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.geobox.extent)

    mask = vector.rasterize(like=raster)

    assert mask.dtype == np.dtype("bool")
    assert mask.values.all()


def test_rasterize_later_rows_win_on_overlap(raster) -> None:
    bounds = raster.gs.geobox.extent.geom
    vector = GeoVector(
        gpd.GeoDataFrame(
            {"class_id": [1, 2]},
            geometry=[bounds, bounds],
            crs=raster.gs.crs,
        )
    )

    result = vector.rasterize(like=raster, column="class_id", dtype="uint8")

    assert (result.values == 2).all()


def test_rasterize_rejects_values_outside_dtype(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.geobox.extent, class_id=300)

    with pytest.raises(ValueError, match="uint8"):
        vector.rasterize(like=raster, column="class_id", dtype="uint8")


def test_rasterize_rejects_non_numeric_property(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.geobox.extent, label="forest")

    with pytest.raises(ValueError, match="numeric raster dtype"):
        vector.rasterize(like=raster, column="label")


def test_rasterize_empty_vector_returns_only_fill(raster) -> None:
    empty = GeoVector.empty(raster.gs.crs)

    result = empty.rasterize(like=raster, fill=0)

    assert not result.values.any()


def test_catalog_register_query_upsert_and_relocate(tmp_path, raster) -> None:
    root = tmp_path / "dataset"
    asset = root / "rasters" / "prediction.zarr"
    asset.mkdir(parents=True)
    labels = GeoVector(
        gpd.GeoDataFrame(
            {"label": ["inside", "outside"]},
            geometry=[raster.gs.geobox.extent.geom, box(0, 0, 1, 1)],
            crs=raster.gs.crs,
        )
    )
    record = GeoVector.from_xarray(raster, path=asset)
    catalog = GeoVector.concat([labels, record])

    matches = labels.query(raster)
    updated = catalog.upsert(
        GeoVector.from_xarray(raster, path=asset, status="complete"),
        on="path",
    )
    path = updated.to_geoparquet(root / "catalog.parquet")
    reopened = read_vector(path)

    assert list(matches.gdf.label) == ["inside"]
    asset_rows = reopened.gdf.loc[reopened.gdf.path.notna()]
    assert len(asset_rows) == 1
    assert asset_rows.iloc[0].path == asset
