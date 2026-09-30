from pathlib import Path
from typing import Literal

from dask.callbacks import Callback
import geopandas as gpd
import numpy as np
from odc.geo import CRS
from odc.geo.geobox import GeoBox
import pandas as pd
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.tasks.manifest import (
    find_labels,
    read_sample_metadata,
    sample_path,
    write_manifest,
)
from geosave_engine.workflow.tasks.sample import SampleFormat, write_sample


def write_dense_sample(
    path, raw, *, day, format: SampleFormat = "zarr"
):
    optical = raw["optical"][["red"]]
    label = raster(
        {"class": np.full((4, 4), day, dtype="uint8")}, optical.gs.geobox
    ).assign_coords(time=np.datetime64(f"2025-01-{day:02d}"))
    write_sample({"label": label, "optical": optical}, path, format=format)
    return str(path)


@pytest.mark.parametrize("format", ["geotiff", "zarr"])
def test_write_manifest_registers_completed_samples(
    tmp_path, raw, format: Literal["geotiff", "zarr"]
):
    samples = tmp_path / "prepared"
    suffix = ".zarr" if format == "zarr" else ""
    paths = {
        "north/a": write_dense_sample(
            samples / "north" / f"a{suffix}", raw, day=1, format=format
        ),
        "south/b": write_dense_sample(
            samples / "south" / f"b{suffix}", raw, day=2, format=format
        ),
    }
    destination = tmp_path / "prepared" / "manifest.parquet"

    result = write_manifest(paths, destination, format=format)

    assert result == str(destination)
    stored = gpd.read_parquet(destination)
    assert list(stored) == [
        "path",
        "format",
        "start_datetime",
        "end_datetime",
        "grid_crs",
        "grid_height",
        "grid_width",
        "geometry",
    ]
    assert stored.path.tolist() == [
        f"north/a{suffix}",
        f"south/b{suffix}",
    ]
    catalog = io.read_vector(destination)
    assert [Path(path) for path in catalog.gdf.path] == [
        Path(path).resolve() for path in paths.values()
    ]
    assert catalog.gdf.geometry.is_valid.all()
    assert catalog.gdf.start_datetime.notna().all()
    assert catalog.gdf.end_datetime.notna().all()
    assert catalog.gdf["format"].tolist() == [format, format]


def test_write_manifest_preserves_ordered_custom_columns_and_nulls(tmp_path, raw):
    samples = tmp_path / "prepared"
    paths = {
        "north/a": write_dense_sample(samples / "north/a.zarr", raw, day=1),
        "south/b": write_dense_sample(samples / "south/b.zarr", raw, day=2),
    }
    metadata = {
        "north/a": {
            "split": "train",
            "quality": 0.9,
            "approved": True,
            "note": None,
        },
        "south/b": {
            "split": "validation",
            "quality": 0.8,
            "approved": False,
            "note": "review",
        },
    }
    destination = samples / "manifest.parquet"

    write_manifest(paths, destination, format="zarr", metadata=metadata)

    stored = gpd.read_parquet(destination)
    assert list(stored) == [
        "path",
        "format",
        "start_datetime",
        "end_datetime",
        "grid_crs",
        "grid_height",
        "grid_width",
        "split",
        "quality",
        "approved",
        "note",
        "geometry",
    ]
    assert stored.split.tolist() == ["train", "validation"]
    assert stored.quality.tolist() == [0.9, 0.8]
    assert stored.approved.tolist() == [True, False]
    assert stored.note.isna().tolist() == [True, False]


def test_write_manifest_accepts_metadata_named_like_factory_parameters(tmp_path, raw):
    sample = write_dense_sample(tmp_path / "samples/a.zarr", raw, day=1)
    destination = tmp_path / "manifest.parquet"

    write_manifest(
        {"a": sample},
        destination,
        format="zarr",
        metadata={
            "a": {
                "crs": "source-crs",
                "fields": "survey-fields",
                "data": "source-data",
            }
        },
    )

    stored = gpd.read_parquet(destination)
    assert stored.loc[0, ["crs", "fields", "data"]].to_dict() == {
        "crs": "source-crs",
        "fields": "survey-fields",
        "data": "source-data",
    }


def test_write_manifest_does_not_compute_sample_pixels(tmp_path, raw):
    sample = write_dense_sample(tmp_path / "samples" / "a.zarr", raw, day=1)
    started = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        write_manifest(
            {"a": sample}, tmp_path / "manifest.parquet", format="zarr"
        )

    assert started == []


def test_write_manifest_replaces_rows_atomically(tmp_path, raw):
    samples = tmp_path / "samples"
    first = write_dense_sample(samples / "a.zarr", raw, day=1)
    second = write_dense_sample(samples / "b.zarr", raw, day=2)
    destination = tmp_path / "manifest.parquet"
    write_manifest({"a": first, "b": second}, destination, format="zarr")

    write_manifest({"b": second}, destination, format="zarr")

    assert io.read_vector(destination).gdf.path.tolist() == [Path(second).resolve()]
    assert not (tmp_path / ".manifest.staging.parquet").exists()


def test_write_manifest_resolves_relative_sample_paths(tmp_path, raw, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = write_dense_sample(Path("prepared/a.zarr"), raw, day=1)
    destination = Path("prepared/manifest.parquet")

    write_manifest({"a": sample}, destination, format="zarr")

    assert gpd.read_parquet(destination).iloc[0].path == "a.zarr"
    assert io.read_vector(destination).gdf.iloc[0].path == (
        tmp_path / "prepared/a.zarr"
    )


def test_write_manifest_combines_samples_from_different_native_crss(tmp_path):
    paths = {}
    grid_crss = []
    for sample_id, crs in (("utm", "EPSG:32633"), ("global", "EPSG:4326")):
        geobox = GeoBox.from_bbox((0, 0, 4, 4), crs=crs, shape=(4, 4))
        grid_crss.append(geobox.crs)
        label = raster(
            {"class": np.ones((4, 4), dtype="uint8")}, geobox
        ).assign_coords(time=np.datetime64("2025-01-01"))
        path = tmp_path / "samples" / f"{sample_id}.zarr"
        paths[sample_id] = write_sample(
            {"label": label, "optical": label.rename({"class": "red"})},
            path,
            format="zarr",
        )

    write_manifest(paths, tmp_path / "manifest.parquet", format="zarr")

    catalog = io.read_vector(tmp_path / "manifest.parquet")
    assert catalog.crs.to_epsg() == 4326
    assert [CRS(value) for value in catalog.gdf.grid_crs] == grid_crss


def test_find_labels_preserves_tree_and_removes_only_the_final_suffix(
    tmp_path,
) -> None:
    labels = tmp_path / "labels"
    path = labels / "train" / "region" / "tile.v1.tif"
    path.parent.mkdir(parents=True)
    path.touch()

    assert find_labels(labels, "**/*.tif") == {
        "train/region/tile.v1": path
    }


@pytest.mark.parametrize(
    ("format", "relative"),
    [
        ("geotiff", "train/region/tile.v1"),
        ("zarr", "train/region/tile.v1.zarr"),
    ],
)
def test_sample_path_preserves_the_suffix_free_identity(
    tmp_path, format, relative
) -> None:
    assert sample_path(
        tmp_path / "prepared", "train/region/tile.v1", format
    ) == tmp_path / "prepared" / relative


def _labels(root: Path) -> dict[str, Path]:
    labels = {
        "train/a": root / "train/a.tif",
        "val/b": root / "val/b.tif",
    }
    for path in labels.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return labels


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label_path": ["../labels/train/a.tif", "../labels/val/b.tif"],
            "split": ["train", "validation"],
            "quality": [0.9, 0.8],
        }
    )


@pytest.mark.parametrize("suffix", [".csv", ".CSV", ".tsv", ".parquet", ".xlsx"])
def test_read_sample_metadata_supports_table_formats_relative_to_the_table(
    tmp_path: Path, suffix: str
) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / "tables" / f"samples{suffix}"
    source.parent.mkdir()
    table = _table()
    if suffix.lower() == ".csv":
        table.to_csv(source, index=False)
    elif suffix == ".tsv":
        table.to_csv(source, sep="\t", index=False)
    elif suffix == ".parquet":
        table.to_parquet(source, index=False)
    else:
        table.to_excel(source, index=False)

    assert read_sample_metadata(source, labels) == {
        "train/a": {"split": "train", "quality": 0.9},
        "val/b": {"split": "validation", "quality": 0.8},
    }


def test_read_sample_metadata_reads_only_the_first_excel_worksheet(
    tmp_path: Path,
) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / "tables" / "samples.xlsx"
    source.parent.mkdir()
    with pd.ExcelWriter(source) as workbook:
        _table().to_excel(workbook, sheet_name="samples", index=False)
        pd.DataFrame(
            {"label_path": ["missing.tif"], "split": ["wrong"]}
        ).to_excel(workbook, sheet_name="ignored", index=False)

    assert read_sample_metadata(source, labels)["train/a"]["split"] == "train"


def test_read_sample_metadata_without_a_table_returns_empty_properties(
    tmp_path: Path,
) -> None:
    labels = _labels(tmp_path / "labels")

    assert read_sample_metadata(None, labels) == {"train/a": {}, "val/b": {}}


def test_read_sample_metadata_rejects_an_unknown_table_format(
    tmp_path: Path,
) -> None:
    labels = _labels(tmp_path / "labels")

    with pytest.raises(ValueError, match=r"\.csv.*\.tsv.*\.parquet.*\.xlsx"):
        read_sample_metadata(tmp_path / "samples.json", labels)


def test_read_sample_metadata_requires_label_path(tmp_path: Path) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / "samples.csv"
    pd.DataFrame({"split": ["train", "validation"]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="label_path"):
        read_sample_metadata(source, labels)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "null"),
        (7, "string"),
        ("/absolute/label.tif", "relative"),
    ],
)
def test_read_sample_metadata_rejects_invalid_label_paths(
    tmp_path: Path, value: object, message: str
) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / ("samples.xlsx" if value == 7 else "samples.parquet")
    table = pd.DataFrame(
        {
            "label_path": [value, "labels/val/b.tif"],
            "split": ["train", "validation"],
        }
    )
    if source.suffix == ".xlsx":
        table.to_excel(source, index=False)
    else:
        table.to_parquet(source, index=False)

    with pytest.raises(ValueError, match=message):
        read_sample_metadata(source, labels)


@pytest.mark.parametrize(
    "paths",
    [
        ["labels/train/a.tif", "labels/train/a.tif", "labels/val/b.tif"],
        [
            "labels/train/a.tif",
            "labels/train/../train/a.tif",
            "labels/val/b.tif",
        ],
    ],
)
def test_read_sample_metadata_rejects_paths_resolving_to_one_label(
    tmp_path: Path, paths: list[str]
) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / "samples.csv"
    pd.DataFrame({"label_path": paths}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="duplicate.*train/a.tif"):
        read_sample_metadata(source, labels)


@pytest.mark.parametrize(
    ("paths", "message"),
    [
        (["labels/train/a.tif"], "missing.*val/b.tif"),
        (
            ["labels/train/a.tif", "labels/val/b.tif", "labels/extra.tif"],
            "extra.*extra.tif",
        ),
        ([], "missing.*train/a.tif"),
    ],
)
def test_read_sample_metadata_requires_an_exact_label_set(
    tmp_path: Path, paths: list[str], message: str
) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / "samples.csv"
    pd.DataFrame({"label_path": paths}).to_csv(source, index=False)

    with pytest.raises(ValueError, match=message):
        read_sample_metadata(source, labels)


def test_read_sample_metadata_requires_string_column_names(tmp_path: Path) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / "samples.xlsx"
    pd.DataFrame(
        [
            ["labels/train/a.tif", "train"],
            ["labels/val/b.tif", "validation"],
        ],
        columns=["label_path", 7],
    ).to_excel(source, index=False)

    with pytest.raises(ValueError, match="column.*string.*7"):
        read_sample_metadata(source, labels)


@pytest.mark.parametrize(
    "reserved",
    [
        "path",
        "format",
        "start_datetime",
        "end_datetime",
        "grid_crs",
        "grid_height",
        "grid_width",
        "geometry",
    ],
)
def test_read_sample_metadata_rejects_manifest_owned_columns(
    tmp_path: Path, reserved: str
) -> None:
    labels = _labels(tmp_path / "labels")
    source = tmp_path / "samples.csv"
    table = pd.DataFrame(
        {
            "label_path": ["labels/train/a.tif", "labels/val/b.tif"],
            reserved: ["first", "second"],
        }
    )
    table.to_csv(source, index=False)

    with pytest.raises(ValueError, match=reserved):
        read_sample_metadata(source, labels)
