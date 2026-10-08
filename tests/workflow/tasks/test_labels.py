from pathlib import Path

import numpy as np
import pytest
from odc.geo.geobox import GeoBox
from shapely.geometry import Point

from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata import io
from geosave_engine.workflow.tasks.labels import find_labels, read_labels


def _label(path: Path, *, dated: bool = True) -> Path:
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    label = raster({"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))}, grid)
    if dated:
        label = label.assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return io.geotiff.write_cog(label, path)


def test_find_labels_preserves_tree_and_removes_only_the_final_suffix(
    tmp_path: Path,
) -> None:
    first = _label(tmp_path / "north" / "a.v1.tif")
    second = _label(tmp_path / "south" / "b.tif")

    assert find_labels(tmp_path, "**/*.tif") == {"north/a.v1": first, "south/b": second}


def test_a_label_directory_becomes_one_row_per_label(tmp_path: Path) -> None:
    first = _label(tmp_path / "north" / "a.tif")
    _label(tmp_path / "south" / "b.tif")

    labels = read_labels(tmp_path)

    assert labels["id"].tolist() == ["north/a", "south/b"]
    assert labels.iloc[0]["assets"]["label"]["href"] == str(first)


def test_a_label_table_keeps_its_caller_columns(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    _label(root / "a.tif")
    labels = GeoVector.from_geometry(
        Point(0, 0),
        properties={
            "id": "a",
            "assets": {"label": {"href": str(root / "a.tif")}},
            "quality": 0.97,
            "surveyor": None,
        },
    )

    read = read_labels(labels.gs.to_geoparquet(tmp_path / "labels.parquet"))

    assert read.loc[0, "quality"] == 0.97
    assert read.loc[0, "surveyor"] is None


def test_a_directory_label_without_time_is_named(tmp_path: Path) -> None:
    path = _label(tmp_path / "labels" / "a.tif", dated=False)

    with pytest.raises(ValueError, match=f"Label raster has no time: {path}"):
        read_labels(tmp_path / "labels")


def test_a_directory_needs_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No labels found"):
        read_labels(tmp_path)


def test_a_label_table_names_its_label_asset(tmp_path: Path) -> None:
    plain = GeoVector.from_geometry(Point(0, 0), properties={"id": "a"})
    other = GeoVector.from_geometry(
        Point(0, 0), properties={"id": "a", "assets": {"image": {"href": "a.tif"}}}
    )

    with pytest.raises(ValueError, match=r"no \['assets'\] column"):
        read_labels(plain.gs.to_geoparquet(tmp_path / "plain.parquet"))
    with pytest.raises(ValueError, match=r"no 'label' asset: \['a'\]"):
        read_labels(other.gs.to_geoparquet(tmp_path / "other.parquet"))


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["a", "a"], "unique"),
        (["a", None], "null"),
        ([1, 2], "text"),
        (["a", "../b"], "inside the output"),
        (["a", "/b"], "inside the output"),
        (["a", ""], "canonical"),
        (["a", "."], "canonical"),
        (["a/b", "a/./b"], "canonical"),
        (["a/b", "a//b"], "canonical"),
        (["a", "a/"], "canonical"),
    ],
)
def test_a_label_table_needs_ids_that_name_one_sample_each(
    tmp_path: Path, ids: list, message: str
) -> None:
    root = tmp_path / "labels"
    _label(root / "a.tif")
    _label(root / "b.tif")
    table = GeoVector.concat(
        [
            GeoVector.from_geometry(
                Point(0, 0),
                properties={
                    "id": name,
                    "assets": {"label": {"href": str(root / f"{name}.tif")}},
                },
            )
            for name in ("a", "b")
        ]
    )
    table["id"] = ids
    path = tmp_path / "labels.parquet"
    table.to_parquet(path)

    with pytest.raises(ValueError, match=message):
        read_labels(path)
