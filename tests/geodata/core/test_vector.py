from datetime import UTC, datetime
from pathlib import Path
import shutil
from typing import cast

import geopandas as gpd
import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from tiler import Tiler
from dask.callbacks import Callback
from pandas import isna
from shapely.geometry import Point, Polygon, box

from geosave_engine.geodata import (
    GeoAnchor,
    GeoVector,
    raster,
    read_raster,
    read_vector,
)
from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.core.stack import stack as build_stack
from geosave_engine.geodata.core.vector import SpatialPredicate
from geosave_engine.geodata.stac.item import item
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


def test_empty_vector_has_a_crs_and_no_footprint() -> None:
    vector = GeoVector.empty("EPSG:4326")

    assert len(vector) == 0
    assert vector.gs.crs.to_epsg() == 4326
    with pytest.raises(ValueError, match="empty"):
        _ = vector.gs.footprint


def test_geometry_properties_become_columns() -> None:
    vector = GeoVector.from_geometry(
        Point(112.15, -8.05), properties={"place_name": "Malang", "split": "train"}
    )

    assert vector.loc[0, "place_name"] == "Malang"
    assert vector.loc[0, "split"] == "train"


def test_a_row_is_read_from_the_files_it_names(tmp_path: Path) -> None:
    path = _written(tmp_path, build_raster(times=1), "scene/image.tif")
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        vector = GeoVector.from_assets(path)

    row = vector.iloc[0]
    assert vector.gs.crs.to_epsg() == 4326
    assert (row.type, row.stac_version) == ("Feature", "1.1.0")
    assert list(row.links) == []
    assert isna(row.datetime)
    assert str(row.start_datetime) == "2025-06-01 00:00:00+00:00"
    assert row.start_datetime.tzinfo is UTC
    assert list(row.assets) == ["image"]
    assert row.assets["image"]["href"] == str(path)
    assert row.assets["image"]["type"].startswith("image/tiff")
    assert row.assets["image"]["bands"] == [{"name": "red"}, {"name": "nir"}]
    assert row["proj:code"] == "EPSG:32749"
    assert row.sources is None
    assert list(vector)[:5] == [
        "id",
        "type",
        "stac_version",
        "stac_extensions",
        "links",
    ]
    assert started == []


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_a_row_keeps_no_raster_open(tmp_path: Path) -> None:
    import gc
    import os

    def open_rasters() -> list[str]:
        links = (Path("/proc/self/fd") / name for name in os.listdir("/proc/self/fd"))
        targets = (os.readlink(link) for link in links if link.is_symlink())
        return sorted(target for target in targets if target.endswith(".tif"))

    path = _written(tmp_path, build_raster(times=1))
    gc.collect()
    before = open_rasters()

    row = GeoVector.from_assets(path)
    # A lazily read raster frees its file once collected; the row must not hold it.
    gc.collect()

    assert len(row) == 1
    assert open_rasters() == before


def test_a_file_that_does_not_exist_cannot_be_registered(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        GeoVector.from_assets(tmp_path / "missing.tif")


def test_layers_become_assets_keyed_by_name(tmp_path: Path) -> None:
    raster = build_raster(times=1)
    optical = _written(tmp_path, raster[["red"]], "optical.tif")
    label = {
        "href": _written(tmp_path, raster[["nir"]], "label.tif"),
        "roles": ["labels"],
    }

    row = GeoVector.from_assets({"optical": optical, "label": label}, id="plot-7").iloc[
        0
    ]

    assert row.id == "plot-7"
    assert row.assets["optical"]["bands"] == [{"name": "red"}]
    assert row.assets["label"]["roles"] == ["labels"]
    assert row.assets["label"]["bands"] == [{"name": "nir"}]


def test_layers_on_different_grids_are_refused(tmp_path: Path) -> None:
    raster = build_raster(times=1)
    whole = _written(tmp_path, raster[["red"]], "whole.tif")
    part = _written(tmp_path, raster[["nir"]].isel(x=slice(0, 1)), "part.tif")

    with pytest.raises(ValueError, match="grid"):
        GeoVector.from_assets({"optical": whole, "label": part})


def test_a_timeless_raster_needs_a_datetime(tmp_path: Path) -> None:
    path = _written(tmp_path, build_raster())

    with pytest.raises(ValueError, match="datetime="):
        GeoVector.from_assets(path)
    row = GeoVector.from_assets(
        path, datetime=datetime(2025, 6, 1, 10, 30, tzinfo=UTC)
    ).iloc[0]

    assert str(row.datetime) == "2025-06-01 10:30:00+00:00"
    assert isna(row.start_datetime) and isna(row.end_datetime)


def test_a_row_keeps_semantic_geometry_and_caller_columns(tmp_path: Path) -> None:
    semantic = box(112.0, -8.2, 112.2, -8.0)

    row = GeoVector.from_assets(
        _written(tmp_path, build_raster(times=1)),
        geometry=semantic,
        properties={"model": "forest-v1", "crs": "caller"},
    ).iloc[0]

    assert row.geometry.equals(semantic)
    assert row.model == "forest-v1"
    assert row.crs == "caller"


def test_a_caller_column_cannot_replace_an_item_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_datetime"):
        GeoVector.from_assets(
            _written(tmp_path, build_raster(times=1)),
            properties={"start_datetime": "caller"},
        )


def test_provider_items_become_the_rows_sources(tmp_path: Path) -> None:
    from geosave_engine.geodata.attrs.models.stac import StacItem, StacMetadata

    loaded = StacMetadata(
        stac_items=(
            StacItem(
                id="S2A_01",
                datetime=datetime(2025, 6, 1, 2, 30),
                properties={"eo:cloud_cover": 10.0},
            ),
        )
    )
    path = _written(tmp_path, build_raster(times=1).gs.rebase(loaded))

    written = GeoVector.from_assets(path).gs.to_geoparquet(tmp_path / "m.parquet")
    row = read_vector(written).iloc[0]

    assert [dict(entry) for entry in row.sources] == [
        {"datetime": "2025-06-01T02:30:00", "eo:cloud_cover": 10.0, "id": "S2A_01"}
    ]


def test_a_read_raster_registers_itself(tmp_path: Path) -> None:
    path = _written(tmp_path, build_raster(times=1), "scene.tif")

    found = GeoVector.from_xarray(read_raster(path, chunks="auto"))

    assert found.loc[0, "assets"]["scene"]["href"] == str(path)
    assert found.loc[0, "id"] == GeoVector.from_assets(path).loc[0, "id"]


def test_a_stack_registers_one_asset_per_group(tmp_path: Path) -> None:
    raster = build_raster(times=1)
    label = _written(tmp_path, raster[["nir"]], "label.tif")
    optical = _written(tmp_path, raster[["red"]], "optical.zarr")
    sample = build_stack(
        {"label": read_raster(label), "sentinel_2_l2a": read_raster(optical)}
    )

    row = GeoVector.from_xarray(sample).iloc[0]

    assert list(row.assets) == ["label", "sentinel_2_l2a"]
    assert row.assets["sentinel_2_l2a"]["href"] == str(optical)


def test_an_object_never_written_must_be_saved_first() -> None:
    with pytest.raises(ValueError, match="from_assets"):
        GeoVector.from_xarray(build_raster(times=1))


@pytest.mark.parametrize("change", ["mask", "to_nan", "unpack", "crop"])
def test_an_object_geosave_changed_must_be_saved_first(
    tmp_path: Path, change: str
) -> None:
    stored = build_raster(times=1, packed=change == "unpack")
    if change != "unpack":
        stored = stored.gs.write_nodata(0)
    read = read_raster(_written(tmp_path, stored, "scene.zarr"), chunks="auto")
    assert GeoVector.from_xarray(read).loc[0, "assets"]["scene"]["href"]
    if change == "mask":
        changed = read.gs.mask(np.array([[True, False], [True, True]]))
    elif change == "to_nan":
        changed = read.gs.to_nan()
    elif change == "unpack":
        changed = read.gs.to_nan().gs.unpack()
    else:
        left, bottom, right, top = read.gs.bounds.bbox
        half = gpd.GeoDataFrame(
            geometry=[box(left, bottom, (left + right) / 2, top)], crs=read.gs.crs
        )
        changed = read.gs.crop(half, mask=False)

    with pytest.raises(ValueError, match="from_assets"):
        GeoVector.from_xarray(changed)


def test_a_tree_is_written_read_and_registered_by_its_directory(tmp_path: Path) -> None:
    from geosave_engine.geodata import write_tree

    cube = build_raster(times=2)

    root = write_tree(cube, tmp_path / "optical", split_bands=True)
    read = read_raster(root, chunks="auto")
    row = GeoVector.from_xarray(read).iloc[0]

    assert root == tmp_path / "optical"
    assert set(read.data_vars) == {"red", "nir"}
    assert read.sizes["time"] == 2
    assert read.encoding["source"] == str(root)
    assert row.assets["optical"]["href"] == str(root)
    # A tree reads its leaves in path order, so split bands come back by name.
    assert row.assets["optical"]["bands"] == [{"name": "nir"}, {"name": "red"}]


def test_a_read_geotiff_names_the_file_it_came_from(tmp_path: Path) -> None:
    path = _written(tmp_path, build_raster(), "scene.tif")

    assert read_raster(path).encoding["source"] == str(path)


def test_a_relative_path_still_opens_from_a_manifest_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _written(tmp_path, build_raster(times=1), "scene.tif")
    monkeypatch.chdir(tmp_path)

    by_path = GeoVector.from_assets("scene.tif")
    by_object = GeoVector.from_xarray(read_raster("scene.tif", chunks="auto"))
    (tmp_path / "catalogs").mkdir()
    written = by_path.gs.to_geoparquet(tmp_path / "catalogs" / "cat.parquet")

    assert by_object.loc[0, "assets"]["scene"]["href"] == str(tmp_path / "scene.tif")
    assert read_vector(written).iloc[0].gs.to_xarray().gs.groups == ("scene",)


@pytest.mark.parametrize("change", ["isel", "time", "subset", "rename", "band"])
def test_an_object_that_no_longer_matches_its_file_must_be_saved_first(
    tmp_path: Path, change: str
) -> None:
    read = read_raster(_written(tmp_path, build_raster(times=2), "scene.zarr"))
    changed = {
        "isel": lambda: read.isel(x=slice(0, 1)),
        "time": lambda: read.isel(time=[0]),
        "subset": lambda: read[["red"]],
        "rename": lambda: read.rename(red="r"),
        "band": lambda: read["red"],
    }[change]()

    with pytest.raises(ValueError, match="from_assets"):
        GeoVector.from_xarray(changed)


def test_a_cropped_band_read_from_geotiff_must_be_saved_first(tmp_path: Path) -> None:
    band = read_raster(_written(tmp_path, build_raster(times=1), "scene.tif"))["red"]

    with pytest.raises(ValueError, match="from_assets"):
        GeoVector.from_xarray(band)


def test_a_multi_group_store_registers_named_group_pointers(tmp_path: Path) -> None:
    from geosave_engine.geodata import read_stack

    raster = build_raster(times=1)
    store = build_stack(
        {"optical": raster[["red"]], "label": raster[["nir"]]}
    ).gs.to_zarr(tmp_path / "sample.zarr")
    record = GeoVector.from_xarray(read_stack(store))
    assert record.iloc[0].assets["optical"]["group"] == "optical"
    assert record.iloc[0].assets["label"]["group"] == "label"


def test_a_tree_in_a_dotted_directory_still_opens(tmp_path: Path) -> None:
    from geosave_engine.geodata import write_tree

    root = write_tree(build_raster(times=2), tmp_path / "s2.v1", split_bands=True)

    assert read_raster(root, chunks="auto").sizes["time"] == 2
    assert GeoVector.from_assets({"optical": root}).loc[0, "assets"]["optical"][
        "href"
    ] == str(root)


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


def _keyed(point: Point, **properties: object) -> gpd.GeoDataFrame:
    return GeoVector.from_geometry(point, properties=properties)


def test_upsert_replaces_caller_key_and_appends_new_key() -> None:
    catalog = GeoVector.concat(
        [
            _keyed(Point(0, 0), record_id="a", status="old"),
            _keyed(Point(1, 1), record_id="b", status="same"),
        ]
    )
    incoming = GeoVector.concat(
        [
            _keyed(Point(2, 2), record_id="a", status="new"),
            _keyed(Point(3, 3), record_id="c", status="new"),
        ]
    )

    result = catalog.gs.upsert(incoming, on="record_id")

    assert list(result.record_id) == ["b", "a", "c"]
    assert list(result.status) == ["same", "new", "new"]


@pytest.mark.parametrize("problem", ["missing", "null", "duplicate"])
def test_upsert_rejects_invalid_incoming_identity(problem: str) -> None:
    existing = _keyed(Point(0, 0), record_id="a")
    if problem == "missing":
        incoming = _keyed(Point(1, 1))
    elif problem == "null":
        incoming = _keyed(Point(1, 1), record_id=None)
    else:
        incoming = GeoVector.concat(
            [_keyed(Point(1, 1), record_id="b"), _keyed(Point(2, 2), record_id="b")]
        )

    with pytest.raises((KeyError, ValueError)):
        existing.gs.upsert(incoming, on="record_id")


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
    result = GeoVector.empty("EPSG:4326").gs.query(_ground(box(0, 0, 1, 1)))

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


def _record(
    href: object, *, id: str | None = None, **properties: object
) -> gpd.GeoDataFrame:
    """Build a STAC-shaped row naming assets that need not exist."""
    assets = href if isinstance(href, dict) else {"prediction": href}
    row = {**item(build_raster(times=1), assets=assets, id=id), **properties}
    frame = gpd.GeoDataFrame(
        {name: [value] for name, value in row.items()},
        geometry=[box(112.0, -8.2, 112.2, -8.0)],
        crs="EPSG:4326",
    )
    for column in ("datetime", "start_datetime", "end_datetime"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _href(vector: gpd.GeoDataFrame) -> str:
    return vector.iloc[0].assets["prediction"]["href"]


def test_asset_href_round_trips_relative_to_a_movable_manifest(tmp_path: Path) -> None:
    original = tmp_path / "original"
    (original / "rasters" / "prediction.zarr").mkdir(parents=True)
    catalog = _record(original / "rasters" / "prediction.zarr")

    catalog.gs.to_geoparquet(original / "catalog.parquet")
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    restored = read_vector(moved / "catalog.parquet")

    assert _href(restored) == str(moved / "rasters" / "prediction.zarr")
    stored = pq.read_table(moved / "catalog.parquet").column("assets")[0].as_py()
    assert stored["prediction"]["href"] == "rasters/prediction.zarr"


def test_absolute_asset_outside_the_manifest_remains_absolute(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "shared" / "prediction.zarr"

    written = _record(outside).gs.to_geoparquet(root / "catalog.parquet")

    assert _href(read_vector(written)) == str(outside)


def test_relative_asset_expands_against_the_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dataset" / "catalogs"
    root.mkdir(parents=True)

    written = _record("../shared/missing.zarr").gs.to_geoparquet(
        root / "catalog.parquet"
    )

    assert _href(read_vector(written)) == str(
        tmp_path / "dataset" / "shared" / "missing.zarr"
    )


def test_a_stac_table_carries_a_bounding_box_and_its_format_key(tmp_path: Path) -> None:
    path = _record("a.zarr").gs.to_geoparquet(tmp_path / "catalog.parquet")

    schema = pq.read_schema(path)

    assert "bbox" in schema.names
    assert orjson.loads(schema.metadata[b"stac-geoparquet"]) == {"version": "1.0.0"}
    restored = read_vector(path)
    assert "bbox" not in restored
    assert restored.gs.to_geoparquet(tmp_path / "again.parquet").is_file()


def test_stac_tools_read_the_table(tmp_path: Path) -> None:
    from pystac import Item
    from stac_geoparquet.arrow import stac_table_to_items

    records = GeoVector.concat(
        [_record("a.zarr", id="a", land_cover="forest"), _record("b.zarr", id="b")]
    )
    path = records.gs.to_geoparquet(tmp_path / "catalog.parquet")

    items = [
        Item.from_dict(entry) for entry in stac_table_to_items(pq.read_table(path))
    ]

    assert [entry.id for entry in items] == ["a", "b"]
    assert items[0].assets["prediction"].href == "a.zarr"
    assert items[0].properties["land_cover"] == "forest"
    assert items[0].properties["proj:code"] == "EPSG:32749"
    assert items[0].bbox is not None
    items[0].validate()


@pytest.mark.parametrize(
    ("ids", "message"), [(["a", None], "null"), (["a", "a"], r"\['a'\]")]
)
def test_a_stac_table_refuses_ids_that_do_not_identify(
    tmp_path: Path, ids: list, message: str
) -> None:
    records = GeoVector.concat([_record("a.zarr"), _record("b.zarr")])
    records["id"] = ids

    with pytest.raises(ValueError, match=message):
        records.gs.to_geoparquet(tmp_path / "catalog.parquet")

    assert not (tmp_path / "catalog.parquet").exists()


def test_a_stac_table_is_written_in_longitude_and_latitude(tmp_path: Path) -> None:
    projected = _record("a.zarr").to_crs("EPSG:32749")

    with pytest.raises(ValueError, match="EPSG:4326"):
        projected.gs.to_geoparquet(tmp_path / "catalog.parquet")


def test_a_row_reads_back_only_the_assets_it_has(tmp_path: Path) -> None:
    both = _record({"image": "a.tif", "label": "b.tif"}, id="a")
    image_only = _record({"image": "c.tif"}, id="b")
    path = GeoVector.concat([both, image_only]).gs.to_geoparquet(
        tmp_path / "catalog.parquet"
    )

    assets = list(read_vector(path).assets)

    assert sorted(assets[0]) == ["image", "label"]
    assert sorted(assets[1]) == ["image"]
    assert assets[0]["label"]["bands"] == [{"name": "red"}, {"name": "nir"}]


def test_an_opaque_asset_pointer_fails_before_writing(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="asset href must be string or path-like"):
        _record(object()).gs.to_geoparquet(tmp_path / "catalog.parquet")

    assert not (tmp_path / "catalog.parquet").exists()


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
        nodata=0,
    )

    vector = GeoVector.vectorize(flags, value_name="class_id")

    assert list(vector.class_id) == [1, 1, 2]
    assert vector.gs.crs == flags.gs.crs
    assert vector.geometry.is_valid.all()


def test_vectorize_preserves_the_value_schema_when_every_pixel_is_nodata(
    raster,
) -> None:
    flags = array(
        np.zeros(raster.gs.geobox.shape, dtype="uint8"),
        raster.gs.geobox,
        nodata=0,
    )

    vector = GeoVector.vectorize(flags, value_name="class_id")
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
    through_method = GeoVector.vectorize(flags)

    assert direct.equals(through_method)


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
    empty = GeoVector.empty(raster.gs.crs)

    result = empty.gs.rasterize(like=raster, fill=0)

    assert not result.values.any()


def test_catalog_register_query_upsert_and_relocate(tmp_path, raster) -> None:
    root = tmp_path / "dataset"
    dated = raster.expand_dims(time=[np.datetime64("2025-06-01", "ns")])
    asset = _written(root, dated, "rasters/prediction.zarr")
    labels = gpd.GeoDataFrame(
        {"label": ["inside", "outside"]},
        geometry=[raster.gs.geobox.extent.geom, box(0, 0, 1, 1)],
        crs=raster.gs.crs,
    )
    catalog = GeoVector.from_assets({"prediction": asset}, id="tile")

    matches = labels.gs.query(read_raster(asset))
    updated = catalog.gs.upsert(
        GeoVector.from_assets(
            {"prediction": asset}, id="tile", properties={"status": "complete"}
        ),
        on="id",
    )
    path = updated.gs.to_geoparquet(root / "catalog.parquet")
    reopened = read_vector(path)

    assert list(matches.label) == ["inside"]
    assert len(reopened) == 1
    assert reopened.iloc[0].status == "complete"
    assert reopened.iloc[0].assets["prediction"]["href"] == str(asset)
    assert len(reopened.gs.query(read_raster(asset))) == 1


@pytest.mark.parametrize("halo", [False, True])
@pytest.mark.parametrize("kind", ["rotated", "wkt", "plain", "mixed"])
def test_tile_reference_keeps_native_identity_and_exact_grid(kind, halo):
    import dask.array as da
    from affine import Affine
    from geopandas.testing import assert_geodataframe_equal
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords
    from tiler import Tiler

    crs = (
        "EPSG:32748"
        if kind != "wkt"
        else "+proj=aeqd +lat_0=17.123 +lon_0=42.456 +datum=WGS84 +units=m +no_defs"
    )
    grid = GeoBox((13, 19), Affine(10, 2, 600000, 1, -10, 9000000), crs)
    geo = xr.DataArray(
        da.ones((13, 19), chunks=(4, 5)), dims=("y", "x"), coords=xr_coords(grid)
    )
    plain = xr.DataArray(da.ones((13, 19), chunks=(4, 5)), dims=("y", "x"))
    parents = {
        "a": plain if kind == "plain" else geo,
        "b": plain if kind in ("plain", "mixed") else geo,
    }
    layouts, padding = {}, {}
    for key, parent in parents.items():
        layout = Tiler(parent.shape, (6, 8), overlap=2, mode="reflect")
        padding[key] = [(0, 0), (0, 0)]
        if halo:
            shape, padding[key] = layout.calculate_padding()
            layout.recalculate(data_shape=shape)
        layouts[key] = layout
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        reference = GeoVector.from_layouts(parents, layouts, padding=padding)
    assert tasks == []
    assert reference.id.is_unique
    assert len(reference) == sum(map(len, layouts.values()))
    for _, row in reference.iterrows():
        near, far = layouts[row.parent_id].get_tile_bbox(int(row.tile_id))
        expected_offset = tuple(
            int(start) - int(width[0])
            for start, width in zip(near, padding[row.parent_id], strict=True)
        )
        assert (row.row_off, row.col_off) == expected_offset
        assert (row.height, row.width) == tuple(far - near)
        parent_grid = parents[row.parent_id].odc.geobox
        if parent_grid is None:
            assert row.geometry is None
            assert all(
                row[name] is None
                for name in ("proj:shape", "proj:transform", "proj:code", "proj:wkt2")
            )
        else:
            code = row["proj:code"] or row["proj:wkt2"]
            recovered = GeoBox(row["proj:shape"], Affine(*row["proj:transform"]), code)
            assert recovered == parent_grid.translate_pix(
                row.col_off, row.row_off
            ).crop((row.height, row.width))
    reordered = GeoVector.from_layouts(
        dict(reversed(list(parents.items()))), layouts, padding=padding
    )
    assert_geodataframe_equal(
        reference.set_index("id").sort_index(), reordered.set_index("id").sort_index()
    )
    assert (reference.crs is None) == (kind == "plain")


def test_tile_reference_requires_matching_parent_keys():
    from tiler import Tiler

    source = xr.DataArray(np.ones((4, 4)), dims=("y", "x"))
    with pytest.raises(ValueError, match="parent"):
        GeoVector.from_layouts({"a": source}, {"b": Tiler((4, 4), (2, 2))})


def _window_raster(height=6, width=8):
    grid = GeoBox.from_bbox((10, 50, 12, 52), "EPSG:4326", shape=(height, width))
    return raster(
        {"red": np.arange(height * width, dtype="float32").reshape(height, width)}, grid
    )


def test_filtered_reference_reads_by_id_without_loading_other_tiles():
    import dask.array as da
    from dask.callbacks import Callback

    source = _window_raster().chunk({"latitude": 2, "longitude": 2})
    parents = {"scene": source}
    layouts = {"scene": Tiler((6, 8), (2, 2))}
    with Callback(
        pretask=lambda *_: (_ for _ in ()).throw(AssertionError("read pixels at setup"))
    ):
        reference = GeoVector.from_layouts(parents, layouts)
        reference = reference.iloc[[11, 0]]
    tile = reference.set_index("id").loc["scene/tile-11"].gs.crop(source)
    np.testing.assert_array_equal(tile.red.values, [[38, 39], [46, 47]])
    assert isinstance(tile.red.data, da.Array)
    assert tile.gs.geobox == _window_raster().gs.geobox.translate_pix(6, 4).crop((2, 2))


def test_reference_keeps_each_rasters_timestamp_order_without_pixels():
    from dask.callbacks import Callback

    image = _window_raster().expand_dims(
        time=np.array(["2024-12-31T14:00", "2024-01-01T03:00"], dtype="datetime64[m]")
    )
    parent = build_stack({"image": image.chunk(), "label": _window_raster().chunk()})
    with Callback(
        pretask=lambda *_: (_ for _ in ()).throw(AssertionError("read pixels at setup"))
    ):
        reference = GeoVector.from_layouts(
            {"scene": parent}, {"scene": Tiler((6, 8), (2, 2))}
        )
    metadata = reference.iloc[0].raster_metadata
    assert metadata["image"]["times"] == ["2024-12-31T14:00:00", "2024-01-01T03:00:00"]
    assert metadata["image"]["bands"] == ["red"]
    assert metadata["label"]["times"] == []


def test_indexed_merger_reassembles_shuffled_predictions():
    from tiler import Merger

    parent = _window_raster()
    parents = {"scene": parent}
    layouts = {"scene": Tiler((6, 8), (4, 4), overlap=2)}
    reference = GeoVector.from_layouts(parents, layouts)
    reference = reference.set_index("id", drop=False)
    merger = Merger(layouts["scene"], logits=1, save_visits=False)
    for sample_id in reversed(reference.id.tolist()):
        row = reference.loc[sample_id]
        tile = row.gs.crop(parent)
        merger.add(int(row.tile_id), tile.red.values[None] * 2)
    np.testing.assert_allclose(merger.merge(), parent.red.values[None] * 2)


def test_regular_and_windowed_asset_rows_share_the_read_path(tmp_path):
    from datetime import datetime
    from geosave_engine.geodata.io import geotiff

    source = raster(
        {"red": np.arange(48, dtype="float32").reshape(6, 8)},
        GeoBox.from_bbox(
            (500000, 9000000, 500080, 9000060), "EPSG:32748", resolution=10
        ),
    )
    path = geotiff.write_cog(source, tmp_path / "image.tif")
    catalog = GeoVector.from_assets(
        {"image": path}, id="scene", datetime=datetime(2024, 1, 1)
    )
    assert catalog.iloc[0].raster_metadata["image"]["bands"] == ["red"]
    whole = catalog.iloc[0].gs.to_xarray()
    reference = GeoVector.from_layouts(
        {"scene": whole}, {"scene": Tiler((6, 8), (2, 2))}
    )
    reference["assets"] = [catalog.iloc[0].assets] * len(reference)
    tile = reference.set_index("id").loc["scene/tile-11"].gs.to_xarray()
    np.testing.assert_array_equal(
        tile.gs.rasters["image"].red.values, [[38, 39], [46, 47]]
    )


def test_plain_raster_rows_read_windows_without_inventing_a_crs():
    import xarray as xr

    source = xr.Dataset({"label": (("y", "x"), np.arange(12).reshape(3, 4))})
    reference = GeoVector.from_layouts(
        {"scene": source}, {"scene": Tiler((3, 4), (2, 2), mode="edge")}
    )
    tile = reference.set_index("id").loc["scene/tile-3"].gs.crop(source)
    np.testing.assert_array_equal(tile.label.values, [[10, 11], [10, 11]])
    assert tile.gs.geobox is None


def test_persisted_metadata_and_window_keep_row_context(tmp_path):
    from geosave_engine.geodata.io import geoparquet
    from geosave_engine.model.encoder import prithvi
    import torch

    image = _window_raster().expand_dims(
        time=np.array(["2024-12-31", "2024-01-01"], dtype="datetime64[D]")
    )
    reference = GeoVector.from_layouts(
        {"scene": image}, {"scene": Tiler((6, 8), (2, 2))}
    )
    path = geoparquet.write(reference, tmp_path / "tiles.parquet", index=False)
    restored = geoparquet.read(path).sample(frac=1, random_state=4)
    row = restored.set_index("id", drop=False).loc["scene/tile-11"]
    context = prithvi.model_context(row)
    torch.testing.assert_close(
        context["temporal_coords"], torch.tensor([[2024.0, 365.0], [2024.0, 0.0]])
    )
    tile = row.gs.crop(image)
    np.testing.assert_array_equal(tile.red.isel(time=0).values, [[38, 39], [46, 47]])


@pytest.mark.parametrize("mode", ["reflect", "wrap"])
@pytest.mark.parametrize("halo", [False, True])
@pytest.mark.parametrize("georeferenced", [False, True])
def test_large_lazy_padding_matches_native_tiler(mode, halo, georeferenced):
    import xarray as xr
    from dask.callbacks import Callback

    source = (
        _window_raster(3, 4)
        if georeferenced
        else xr.Dataset({"red": (("y", "x"), np.arange(12).reshape(3, 4))})
    )
    original = source.red.values
    layout = Tiler((3, 4), (16, 16), overlap=2, mode=mode)
    padding = [(0, 0), (0, 0)]
    if halo:
        shape, padding = layout.calculate_padding()
        layout.recalculate(data_shape=shape)
    # Native Tiler may produce no tiles without a halo for a very small scene;
    # use a larger source to exercise repeated fringe padding in that case.
    if not halo:
        source = (
            _window_raster(13, 19)
            if georeferenced
            else xr.Dataset({"red": (("y", "x"), np.arange(247).reshape(13, 19))})
        )
        original = source.red.values
        layout = Tiler((13, 19), (16, 16), overlap=2, mode=mode)
    reference = GeoVector.from_layouts(
        {"scene": source}, {"scene": layout}, padding={"scene": padding}
    )
    padded = np.pad(original, padding, mode=mode)
    for row in reference.itertuples():
        with Callback(pretask=lambda *_: pytest.fail("window read computed pixels")):
            tile = reference.set_index("id").loc[row.id].gs.crop(source.chunk())
        assert tile.red.shape == (16, 16)
        np.testing.assert_array_equal(
            tile.red.values, layout.get_tile(padded, row.tile_id)
        )
