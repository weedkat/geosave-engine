from geosave_engine.geodata import cuts
from pathlib import Path
from uuid import uuid4

import fsspec
import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point, box

from geosave_engine.geodata import GeoVector, read_vector


@pytest.mark.parametrize("kind", ["geo", "plain", "mixed", "wkt"])
def test_tile_reference_round_trip_preserves_exact_grids(tmp_path, kind):
    import numpy as np
    import xarray as xr
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    from geosave_engine.geodata.io import geoparquet
    from tests.geodata.conftest import whole_windows

    crs = (
        "EPSG:32748"
        if kind != "wkt"
        else "+proj=aeqd +lat_0=17.123 +lon_0=42.456 +datum=WGS84 +units=m +no_defs"
    )
    grid = GeoBox((12, 16), Affine(10, 0, 100, 0, -10, 500), crs)
    geo = xr.Dataset({"B": (("y", "x"), np.ones((12, 16)))}, coords=xr_coords(grid))
    plain = xr.Dataset({"B": (("y", "x"), np.ones((12, 16)))})
    sources = [plain] if kind == "plain" else [geo, plain] if kind == "mixed" else [geo]
    parent_ids = [f"parent-{position}" for position in range(len(sources))]
    parents = dict(zip(parent_ids, sources, strict=True))
    reference = cuts.chips(whole_windows(parents), (6, 8), overlap=2)
    path = geoparquet.write(reference, tmp_path / "reference.parquet", index=False)
    restored = geoparquet.read(path).sample(frac=1, random_state=41)
    lookup = restored.set_index("id", verify_integrity=True)

    assert set(lookup.index) == set(reference.id)
    assert restored.crs == reference.crs
    for sample_id in reference.id:
        row = lookup.loc[sample_id]
        actual = parents[row.parent].gs.geobox
        assert row.chip == reference.set_index("id").loc[sample_id, "chip"]
        if actual is not None:
            actual = actual.translate_pix(row.col_off, row.row_off).crop(
                (row.height, row.width)
            )
        if actual is None:
            assert row.geometry is None
            assert row["crs"] is None
        else:
            recovered = GeoBox(
                (row.height, row.width), Affine(*row["transform"]), row["crs"]
            )
            assert recovered == actual
            if kind == "wkt":
                assert not row["crs"].startswith("EPSG:")


def test_geoparquet_catalog_supports_bbox_and_column_filtering(
    tmp_path: Path,
) -> None:
    vector = gpd.GeoDataFrame(
        {"name": ["near", "far"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    path = vector.gs.to_geoparquet(
        tmp_path / "catalog.parquet", write_covering_bbox=True
    )

    selected = read_vector(path, bbox=(-1, -1, 2, 2), columns=["name", "geometry"])

    assert list(selected.name) == ["near"]


def test_geoparquet_does_not_write_covering_bbox_by_default(tmp_path: Path) -> None:
    path = GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(
        tmp_path / "catalog.parquet"
    )

    assert "bbox" not in pq.read_schema(path).names


def test_geoparquet_write_replaces_stale_staging_file(tmp_path: Path) -> None:
    target = tmp_path / "catalog.parquet"
    staged = tmp_path / ".catalog.staging.parquet"
    staged.write_text("interrupted")

    GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(target)

    assert target.is_file()
    assert not staged.exists()


def test_empty_geoparquet_catalog_round_trips(tmp_path: Path) -> None:
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))
    path = empty.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = read_vector(path)

    assert len(restored) == 0
    assert restored.crs.to_epsg() == 4326


def memory_catalog() -> str:
    return f"memory://geosave-tests/{uuid4()}/catalog.parquet"


def test_a_row_without_assets_stays_without() -> None:
    destination = memory_catalog()
    vector = gpd.GeoDataFrame(
        {"id": ["a"], "assets": [None]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )

    vector.gs.to_geoparquet(destination)

    assert read_vector(destination).assets.isna().all()


def test_remote_write_refuses_existing_catalog() -> None:
    destination = memory_catalog()
    vector = GeoVector.from_geometry(Point(0, 0))
    vector.gs.to_geoparquet(destination)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        vector.gs.to_geoparquet(destination)


def test_file_url_returns_a_usable_local_path(tmp_path: Path) -> None:
    destination = tmp_path / "catalog.parquet"

    written = GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(
        destination.as_uri()
    )

    assert written == destination
    assert len(read_vector(written)) == 1


def test_invalid_remote_suffix_creates_nothing() -> None:
    destination = memory_catalog().replace(".parquet", ".json")
    filesystem, target = fsspec.core.url_to_fs(destination)

    with pytest.raises(ValueError, match="must end"):
        GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_failed_new_remote_write_uploads_nothing(monkeypatch) -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)

    def fail(_self, path: Path, **options: object) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)

    with pytest.raises(RuntimeError, match="encode failed"):
        GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_failed_remote_overwrite_does_not_delete_existing_object(
    monkeypatch,
) -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)
    filesystem.pipe(target, b"existing")

    def fail(_self, path: Path, **options: object) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)

    with pytest.raises(RuntimeError, match="encode failed"):
        GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(
            destination, overwrite=True
        )

    assert filesystem.cat(target) == b"existing"
