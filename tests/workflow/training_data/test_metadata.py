from pathlib import Path

import pandas as pd
import pytest

from geosave_engine.workflow.training_data.manifest import read_sample_metadata


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
