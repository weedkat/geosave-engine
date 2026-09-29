"""Discover labels, join caller metadata, and publish the sample manifest."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pandas as pd

from geosave_engine.geodata import GeoVector

from .sample import SampleFormat, open_sample

_TABLE_SUFFIXES = (".csv", ".tsv", ".parquet", ".xlsx")
_OWNED_COLUMNS = (
    "path",
    "format",
    "start_datetime",
    "end_datetime",
    "grid_crs",
    "grid_height",
    "grid_width",
    "geometry",
)


def find_labels(root: Path, pattern: str) -> dict[str, Path]:
    """Return sorted label paths keyed by stable relative sample IDs."""
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No labels found for {pattern!r} under {root}")

    labels = {
        path.relative_to(root).with_suffix("").as_posix(): path for path in paths
    }
    if len(labels) != len(paths):
        raise ValueError("Labels must map to unique sample paths")
    return labels


def sample_path(
    root: Path, sample_id: str, format: Literal["geotiff", "zarr"]
) -> Path:
    """Return the format-specific path for one suffix-free sample ID."""
    path = root / sample_id
    if format == "zarr":
        return path.parent / f"{path.name}.zarr"
    return path


def _read_table(source: Path) -> pd.DataFrame:
    """Read one supported metadata table."""
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".tsv":
        return pd.read_csv(source, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".xlsx":
        return pd.read_excel(source, sheet_name=0, engine="openpyxl")
    raise ValueError(
        f"Unsupported metadata table {source}; choose from {list(_TABLE_SUFFIXES)}"
    )


def read_sample_metadata(
    source: str | Path | None, labels: dict[str, Path]
) -> dict[str, dict[str, object]]:
    """Read caller columns keyed by stable sample IDs."""
    if source is None:
        return {key: {} for key in labels}

    table_path = Path(source)
    table = _read_table(table_path)
    invalid_columns = [column for column in table.columns if not isinstance(column, str)]
    if invalid_columns:
        raise ValueError(f"Metadata column names must be strings: {invalid_columns}")
    if "label_path" not in table:
        raise ValueError("Metadata table requires a 'label_path' column")
    if reserved := sorted(set(_OWNED_COLUMNS) & set(table.columns)):
        raise ValueError(f"Metadata columns are owned by the manifest: {reserved}")

    resolved_paths = []
    for value in table["label_path"]:
        if pd.isna(value):
            raise ValueError("Metadata label_path cannot be null")
        if not isinstance(value, str):
            raise ValueError(
                f"Metadata label_path must be a string, got {type(value).__name__}"
            )
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"Metadata label_path must be relative: {value}")
        resolved_paths.append((table_path.parent / relative).resolve())

    duplicates = sorted(
        {path for path in resolved_paths if resolved_paths.count(path) > 1}, key=str
    )
    if duplicates:
        raise ValueError(f"Metadata contains duplicate label_path values: {duplicates}")

    labels_by_path = {path.resolve(): key for key, path in labels.items()}
    discovered_paths = set(labels_by_path)
    supplied_paths = set(resolved_paths)
    if missing := sorted(discovered_paths - supplied_paths, key=str):
        raise ValueError(f"Metadata is missing label paths: {missing}")
    if extra := sorted(supplied_paths - discovered_paths, key=str):
        raise ValueError(f"Metadata contains extra label paths: {extra}")

    caller_columns = [column for column in table.columns if column != "label_path"]
    rows = table[caller_columns].to_dict(orient="records")
    by_path = dict(zip(resolved_paths, rows, strict=True))
    return {key: by_path[path.resolve()] for key, path in labels.items()}


def write_manifest(
    samples: dict[str, str],
    destination: str | Path,
    *,
    format: SampleFormat,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Register completed sample stores and publish their manifest.

    Args:
        samples: Completed sample paths keyed by stable sample ID.
        destination: GeoParquet manifest path.
        format: Persisted sample representation.
        metadata: Caller-owned properties keyed like ``samples``.

    Returns:
        Path to the completed manifest.
    """
    properties = metadata or {}
    records = []
    for sample_key, path in samples.items():
        sample_path = Path(path).resolve()
        with open_sample(sample_path, format=format) as sample:
            anchor = sample.gs.anchor
            records.append(
                GeoVector.from_xarray(
                    sample,
                    crs="EPSG:4326",
                    fields=("time",),
                    path=sample_path,
                    format=format,
                    grid_crs=str(anchor.crs),
                    grid_height=anchor.geobox.height,
                    grid_width=anchor.geobox.width,
                    **properties.get(sample_key, {}),
                )
            )

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    custom_columns = list(
        dict.fromkeys(
            column
            for row in properties.values()
            for column in row
        )
    )
    columns = [*_OWNED_COLUMNS[:-1], *custom_columns, _OWNED_COLUMNS[-1]]
    catalog = GeoVector.concat(records)
    GeoVector(catalog.gdf[columns]).to_geoparquet(output, overwrite=True)
    return str(output)
