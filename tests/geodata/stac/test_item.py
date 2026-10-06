"""Items built from the assets of saved rasters."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from odc.geo.geobox import GeoBox
from stac_geoparquet.arrow import stac_table_to_items

from geosave_engine.geodata import GeoVector, io, raster
from geosave_engine.geodata.stac.asset import from_path
from geosave_engine.geodata.stac.item import from_assets

from tests.geodata.conftest import build_raster

IDS = ["forest_20250601T000000", "forest_20250602T000000"]


@pytest.mark.integration
def test_items_validate_against_the_stac_schemas(tmp_path: Path) -> None:
    for item in build_raster(times=2).gs.to_items(tmp_path / "forest"):
        item.validate()


def test_from_assets_names_layers_and_leaves_them_unowned(tmp_path: Path) -> None:
    optical = from_path(build_raster(times=2).gs.to_zarr(tmp_path / "optical.zarr"))
    label = from_path(
        build_raster(times=1).isel(time=0).gs.to_cog(tmp_path / "label")[0]
    )

    item = from_assets({"optical": optical, "label": label}, id="s0")

    assert item.id == "s0"
    assert list(item.assets) == ["optical", "label"]
    assert item.datetime is None
    assert item.common_metadata.start_datetime == optical.common_metadata.start_datetime
    assert item.common_metadata.end_datetime == optical.common_metadata.end_datetime
    assert optical.owner is None and label.owner is None


def test_from_assets_dates_timeless_assets_explicitly(tmp_path: Path) -> None:
    dem = from_path(build_raster().gs.to_cog(tmp_path / "dem")[0])
    instant = datetime(2020, 1, 1, tzinfo=UTC)

    assert from_assets({"dem": dem}, id="dem", datetime=instant).datetime == instant
    with pytest.raises(ValueError, match="datetime="):
        from_assets({"dem": dem}, id="dem")


def test_a_footprint_is_built_for_a_grid_without_an_epsg_code(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=12.34 +lon_0=56.78 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    label = raster({"class": np.ones((4, 4), dtype="uint8")}, grid).assign_coords(
        time=np.datetime64("2025-01-15T12:00:00")
    )
    path = io.geotiff.write_cog(label, tmp_path / "laea.tif")

    item = from_assets({"label": from_path(path)}, id="laea")

    assert item.bbox[0] == pytest.approx(56.78, abs=0.01)
    assert item.bbox[1] == pytest.approx(12.34, abs=0.01)


def test_items_survive_the_table(tmp_path: Path) -> None:
    items = build_raster(times=2).gs.to_items(tmp_path / "forest")

    table = GeoVector.from_items(items).gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = list(stac_table_to_items(pq.read_table(table)))
    assert [entry["id"] for entry in restored] == IDS
    assert [band["name"] for band in restored[0]["assets"]["red"]["eo:bands"]] == [
        "red"
    ]


def test_an_asset_without_a_grid_cannot_place_an_item() -> None:
    import pystac

    bare = pystac.Asset("s0/notes.txt", roles=["data"])

    with pytest.raises(ValueError, match="states no grid"):
        from_assets({"notes": bare}, id="s0")
