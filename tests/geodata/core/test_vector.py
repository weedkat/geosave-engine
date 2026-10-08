from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback
from shapely.geometry import Point, Polygon, box

from geosave_engine.geodata import (
    GeoAnchor,
    GeoVector,
    read_raster,
)
from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.core.vector import SpatialPredicate
from geosave_engine.geodata.transform.vector import rasterize, vectorize
from geosave_engine.geodata.io import geotiff, zarr

from tests.geodata.conftest import build_raster


def _written(root: Path, raster: xr.Dataset, name: str = "image.tif") -> Path:
    """Write one raster the way a sample layer is written and return its path."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zarr":
        zarr.write(raster, path)
    else:
        flat = raster.squeeze("time", drop=False) if "time" in raster.dims else raster
        geotiff.write_cog(flat, path)
    return path


def _ground(geometry, timespan: str | None = None) -> GeoAnchor:
    """Anchor fitting one longitude/latitude geometry exactly."""
    return GeoAnchor.from_geometry(
        geometry, shape=4, crs="EPSG:4326", anchor="floating", timespan=timespan
    )


def test_geographic_operations_require_a_crs() -> None:
    with pytest.raises(AttributeError, match="GeoDataFrame"):
        _ = pd.DataFrame({"a": [1]}).gs
    with pytest.raises(ValueError, match="CRS"):
        _ = gpd.GeoDataFrame(geometry=[Point(0, 0)]).gs.crs


def test_the_accessor_survives_native_selection_and_concatenation() -> None:
    frame = gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )

    assert frame[frame.name == "b"].gs.crs.to_epsg() == 4326
    assert len(pd.concat([frame, frame]).gs.query(_ground(box(-1, -1, 2, 2)))) == 4


def test_a_renamed_geometry_column_still_answers() -> None:
    frame = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:4326"
    ).rename_geometry("footprint")

    assert frame.gs.footprint.geom.equals(box(0, 0, 2, 2))
    assert list(frame.gs.query(_ground(box(1, 1, 3, 3))).name) == ["a"]
    assert GeoVector.concat([frame, frame]).geometry.name == "geometry"


def test_geometry_properties_become_columns() -> None:
    vector = GeoVector.from_geometry(
        Point(112.15, -8.05), properties={"place_name": "Malang", "split": "train"}
    )

    assert vector.loc[0, "place_name"] == "Malang"
    assert vector.loc[0, "split"] == "train"


def test_a_read_geotiff_names_the_file_it_came_from(tmp_path: Path) -> None:
    path = _written(tmp_path, build_raster(), "scene.tif")

    assert read_raster(path).encoding["source"] == str(path)


def test_query_reads_time_columns_that_carry_no_timezone() -> None:
    table = _dated()
    for column in ("datetime", "start_datetime", "end_datetime"):
        table[column] = table[column].dt.tz_localize(None)

    assert list(table.gs.query(_ground(box(-1, -1, 2, 2), "2025-01")).id) == ["jan"]


def test_concat_unions_columns_resets_index_and_keeps_duplicates() -> None:
    first = GeoVector.from_geometry(Point(0, 0), properties={"name": "first"})
    renamed = GeoVector.from_geometry(
        Point(1, 1), properties={"score": 4}
    ).rename_geometry("footprint")

    combined = GeoVector.concat([first, renamed, first])

    assert list(combined.index) == [0, 1, 2]
    assert list(combined.geometry) == [Point(0, 0), Point(1, 1), Point(0, 0)]
    assert list(combined.name.iloc[[0, 2]]) == ["first", "first"]
    assert combined.loc[1, "score"] == 4


def test_concat_requires_one_explicit_crs_plane() -> None:
    geographic = GeoVector.from_geometry(Point(0, 0))
    projected = GeoVector.from_geometry(Point(0, 0), crs="EPSG:3857")

    with pytest.raises(ValueError, match="to_crs"):
        GeoVector.concat([geographic, projected])


def _places(names: list[str], geometries: list) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"name": names}, geometry=geometries, crs="EPSG:4326")


def test_query_returns_original_intersecting_rows_in_source_order() -> None:
    collection = _places(
        ["left", "middle", "right"],
        [box(0, 0, 2, 2), box(2, 0, 4, 2), box(8, 0, 10, 2)],
    )

    result = collection.gs.query(_ground(box(1, -1, 3, 3)))

    assert list(result.name) == ["left", "middle"]
    assert result.iloc[0].geometry.equals(collection.iloc[0].geometry)


def test_query_predicate_describes_each_row_relative_to_target() -> None:
    collection = _places(
        ["inside", "outside", "container"],
        [box(1, 1, 2, 2), box(5, 5, 6, 6), box(-1, -1, 4, 4)],
    )
    target = _ground(box(0, 0, 3, 3))

    assert list(collection.gs.query(target, predicate="within").name) == ["inside"]
    assert list(collection.gs.query(target, predicate="contains").name) == ["container"]


def test_query_reprojects_the_anchor_onto_the_table() -> None:
    collection = _places(["place"], [box(112.0, -8.2, 112.2, -8.0)])
    anchor = GeoAnchor.from_coordinates(-8.1, 112.1, shape=4, resolution=100)

    result = collection.gs.query(anchor)

    assert anchor.crs.epsg != 4326
    assert list(result.name) == ["place"]
    assert result.gs.crs.to_epsg() == 4326


def test_query_takes_a_raster_by_its_anchor(raster) -> None:
    collection = gpd.GeoDataFrame(
        {"label": ["inside", "outside"]},
        geometry=[raster.gs.geobox.extent.geom, box(0, 0, 1, 1)],
        crs=raster.gs.crs,
    )

    assert list(collection.gs.query(raster).label) == ["inside"]


def test_query_empty_collection_preserves_schema() -> None:
    result = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326")).gs.query(
        _ground(box(0, 0, 1, 1))
    )

    assert len(result) == 0
    assert result.gs.crs.to_epsg() == 4326


def test_query_rejects_unknown_predicate() -> None:
    vector = GeoVector.from_geometry(Point(0, 0))

    with pytest.raises(ValueError, match="predicate"):
        vector.gs.query(
            _ground(box(-1, -1, 1, 1)), predicate=cast("SpatialPredicate", "touch-ish")
        )


def _dated() -> gpd.GeoDataFrame:
    """Three rows on one ground: two spans and one lone instant."""
    table = gpd.GeoDataFrame(
        {
            "id": ["jan", "jun", "instant"],
            "datetime": [None, None, datetime(2025, 6, 12, tzinfo=UTC)],
            "start_datetime": [
                datetime(2025, 1, 10, tzinfo=UTC),
                datetime(2025, 6, 10, tzinfo=UTC),
                None,
            ],
            "end_datetime": [
                datetime(2025, 1, 15, tzinfo=UTC),
                datetime(2025, 6, 15, tzinfo=UTC),
                None,
            ],
        },
        geometry=[box(0, 0, 1, 1)] * 3,
        crs="EPSG:4326",
    )
    for column in ("datetime", "start_datetime", "end_datetime"):
        table[column] = pd.to_datetime(table[column], utc=True)
    return table


def test_query_compares_time_where_the_anchor_and_the_table_state_one() -> None:
    table = _dated()
    ground = box(-1, -1, 2, 2)

    assert list(table.gs.query(_ground(ground)).id) == ["jan", "jun", "instant"]
    assert list(table.gs.query(_ground(ground, "2025-01")).id) == ["jan"]
    assert list(table.gs.query(_ground(ground, "2025-06-12")).id) == ["jun", "instant"]
    assert list(table.gs.query(_ground(ground, "2025-03")).id) == []


def test_query_ignores_time_on_a_table_that_states_none() -> None:
    undated = _places(["place"], [box(0, 0, 1, 1)])

    assert list(undated.gs.query(_ground(box(-1, -1, 2, 2), "2025-01")).name) == [
        "place"
    ]


@pytest.mark.parametrize("problem", ["null", "empty"])
def test_rasterize_refuses_a_missing_geometry(raster, problem: str) -> None:
    geometry = None if problem == "null" else Polygon()
    vector = gpd.GeoDataFrame(geometry=[geometry], crs=raster.gs.crs)

    with pytest.raises(ValueError, match="null, empty, or invalid"):
        vector.gs.rasterize(raster)


def test_rasterize_refuses_a_geometry_naming_no_ground(raster) -> None:
    bowtie = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], crs=raster.gs.crs
    )

    with pytest.raises(ValueError, match="invalid geometry"):
        bowtie.gs.rasterize(raster)


def test_vectorize_creates_one_row_per_connected_flag_region(raster) -> None:
    flags = array(
        np.array([[1, 0, 1], [1, 0, 0], [2, 2, 0]], dtype="uint8"),
        GeoAnchor.from_bbox(
            raster.gs.geobox.boundingbox,
            shape=(3, 3),
            crs=raster.gs.crs,
        ).geobox,
        dims=("y", "x"),
        nodata=0,
    )

    vector = flags.gs.vectorize(value_name="class_id")

    assert list(vector.class_id) == [1, 1, 2]
    assert vector.gs.crs == flags.gs.crs
    assert vector.geometry.is_valid.all()


def test_vectorize_preserves_the_value_schema_when_every_pixel_is_nodata(
    raster,
) -> None:
    flags = array(
        np.zeros(raster.gs.geobox.shape, dtype="uint8"),
        raster.gs.geobox,
        dims=("y", "x"),
        nodata=0,
    )

    vector = flags.gs.vectorize(value_name="class_id")
    restored = vector.gs.rasterize(
        like=flags,
        column="class_id",
        fill=0,
        dtype="uint8",
    )

    assert list(vector) == ["class_id", "geometry"]
    assert vector.class_id.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(restored.values, np.zeros(flags.shape, "uint8"))


def test_vector_method_delegates_to_vector_transform(raster) -> None:
    flags = raster.red.astype("uint8")

    direct = vectorize(flags)
    through_method = flags.gs.vectorize()

    assert direct.equals(through_method)


def test_vectorize_computes_dask_flags_explicitly(raster) -> None:
    flags = raster.red.chunk({"y": 1, "x": 1})
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        flags.gs.vectorize()

    assert started


def test_vectorize_requires_one_spatial_plane(raster) -> None:
    flags = raster.red.expand_dims(time=["2025-01-01"])

    with pytest.raises(ValueError, match="two-dimensional"):
        flags.gs.vectorize()


def test_vectorize_rejects_unsupported_values(raster) -> None:
    flags = raster.red.astype("complex64")

    with pytest.raises(ValueError, match="dtype"):
        flags.gs.vectorize()


def test_vectorize_rejects_same_shape_mask_on_another_grid(raster) -> None:
    shifted = raster.red.assign_coords(x=raster.x + 10)

    with pytest.raises(ValueError, match="exact flags grid"):
        raster.red.gs.vectorize(mask=shifted)


def test_rasterize_values_back_onto_reference_grid(raster) -> None:
    flags = array(
        np.array([[1, 1], [2, 0]], dtype="uint8"),
        raster.gs.geobox,
        dims=("y", "x"),
        nodata=0,
    )
    vector = flags.gs.vectorize(value_name="class_id")

    restored = vector.gs.rasterize(
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
    through_method = vector.gs.rasterize(raster)

    xr.testing.assert_identical(direct, through_method)


def test_rasterize_without_column_returns_boolean_presence(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.geobox.extent)

    mask = vector.gs.rasterize(like=raster)

    assert mask.dtype == np.dtype("bool")
    assert mask.values.all()


def test_rasterize_later_rows_win_on_overlap(raster) -> None:
    bounds = raster.gs.geobox.extent.geom
    vector = gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[bounds, bounds],
        crs=raster.gs.crs,
    )

    result = vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")

    assert (result.values == 2).all()


def test_rasterize_rejects_values_outside_dtype(raster) -> None:
    vector = GeoVector.from_geometry(
        raster.gs.geobox.extent, properties={"class_id": 300}
    )

    with pytest.raises(ValueError, match="uint8"):
        vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")


def test_rasterize_rejects_non_numeric_property(raster) -> None:
    vector = GeoVector.from_geometry(
        raster.gs.geobox.extent, properties={"label": "forest"}
    )

    with pytest.raises(ValueError, match="numeric raster dtype"):
        vector.gs.rasterize(like=raster, column="label")


def test_rasterize_empty_vector_returns_only_fill(raster) -> None:
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=raster.gs.crs))

    result = empty.gs.rasterize(like=raster, fill=0)

    assert not result.values.any()


def test_query_filters_time_on_a_table_with_only_datetime() -> None:
    raster = build_raster(times=2)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    frame = gpd.GeoDataFrame(
        {
            "id": ["first", "second"],
            "datetime": pd.to_datetime(raster.time.values, utc=True),
        },
        geometry=[footprint, footprint],
        crs="EPSG:4326",
    )

    assert frame.gs.query(raster.isel(time=[0]))["id"].tolist() == ["first"]


def test_query_reads_a_span_where_a_row_has_no_datetime() -> None:
    raster = build_raster(times=2)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    frame = gpd.GeoDataFrame(
        {
            "id": ["store", "later"],
            "datetime": pd.to_datetime([None, "2025-07-01"], utc=True),
            "start_datetime": pd.to_datetime(["2025-06-01", None], utc=True),
            "end_datetime": pd.to_datetime(["2025-06-02", None], utc=True),
        },
        geometry=[footprint, footprint],
        crs="EPSG:4326",
    )

    assert frame.gs.query(raster)["id"].tolist() == ["store"]


def test_undated_records_remain_spatially_selectable_in_a_time_query():
    source = GeoVector.from_geometry(box(0, 0, 1, 1))
    source["datetime"] = pd.NaT
    selected = source.gs.query(_ground(box(-1, -1, 2, 2), "2025-01"))
    assert len(selected) == 1


REMOVED = (
    "normalize",
    "timespan",
    "to_anchor",
    "to_items",
    "from_items",
    "upsert",
    "to_raster",
    "to_stack",
    "raster_ids",
    "stack_ids",
    "empty",
    "vectorize",
)


@pytest.mark.parametrize("name", REMOVED)
def test_geovector_is_spatial_only(name: str) -> None:
    assert not hasattr(GeoVector, name)


def test_a_geometry_factory_adds_no_time_column() -> None:
    frame = GeoVector.from_geometry(Point(0, 0), properties={"name": "a"})

    assert list(frame.columns) == ["name", "geometry"]
