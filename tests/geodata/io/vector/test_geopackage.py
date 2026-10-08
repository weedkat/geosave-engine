"""GeoPackage layers read and write through the shared storage writer."""

from __future__ import annotations

from pathlib import Path

import fsspec
import geopandas as gpd
import pytest
from shapely.geometry import Point

from geosave_engine.geodata import read_vector


def _plots() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"class": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )


def test_a_layer_is_named_after_the_file_by_default(tmp_path: Path) -> None:
    path = _plots().gs.to_geopackage(tmp_path / "survey.gpkg")

    assert gpd.list_layers(path)["name"].tolist() == ["survey"]
    assert read_vector(path)["class"].tolist() == [1, 2]


def test_an_existing_geopackage_refuses_without_overwrite(tmp_path: Path) -> None:
    path = _plots().gs.to_geopackage(tmp_path / "survey.gpkg")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _plots().gs.to_geopackage(path)

    _plots().iloc[[0]].gs.to_geopackage(path, overwrite=True)
    assert len(read_vector(path)) == 1


def test_a_geopackage_writes_to_a_remote_url(bucket: str) -> None:
    written = _plots().gs.to_geopackage(f"{bucket}/survey.gpkg")

    assert written == f"{bucket}/survey.gpkg"
    assert fsspec.filesystem("memory").size(written) > 0
