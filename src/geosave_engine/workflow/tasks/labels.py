"""Read a label set as a table of STAC items."""

from pathlib import Path, PurePosixPath

from geosave_engine.geodata import GeoDataFrame, GeoVector, read_raster, read_vector

_TABLE_SUFFIXES = (".parquet", ".geoparquet")


def find_labels(root: Path, pattern: str) -> dict[str, Path]:
    """Return label paths in path order, keyed by sample ID."""
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No labels found for {pattern!r} under {root}")

    label_paths = {
        path.relative_to(root).with_suffix("").as_posix(): path for path in paths
    }
    if len(label_paths) != len(paths):
        raise ValueError("Labels must map to unique sample paths")
    return label_paths


def read_labels(source: str | Path, pattern: str = "**/*.tif") -> GeoDataFrame:
    """Read a label set, indexing a directory of label rasters into a table.

    Args:
        source: GeoParquet STAC table with one row per label, each naming its
            raster as the asset `label`, or a directory of label rasters.
        pattern: Recursive glob selecting the labels of a directory.

    Returns:
        Table with a unique text `id` and a `label` asset for every row,
        beside any caller columns. A sample's time is its label raster's own.

    Raises:
        ValueError: A directory holds no label, two labels share a sample
            path, or a label carries no time; or a table lacks `id` or
            `assets`, a row names no `label` asset, or an id is null, not
            text, repeated, noncanonical, or names a folder outside the output.
    """
    path = Path(source)
    if path.suffix.lower() not in _TABLE_SUFFIXES:
        rows = []
        for sample_id, label_path in find_labels(path, pattern).items():
            with read_raster(label_path) as label:
                if label.gs.timespan is None:
                    raise ValueError(f"Label raster has no time: {label_path}")
            rows.append(GeoVector.from_assets({"label": label_path}, id=sample_id))
        return GeoVector.concat(rows)

    labels = read_vector(path)
    absent = [name for name in ("id", "assets") if name not in labels]
    if absent:
        raise ValueError(f"Label table {path} has no {absent} column")

    ids = labels["id"]
    if bool(ids.isna().any()):
        raise ValueError(f"Label table {path} holds a null id")
    if not all(isinstance(sample_id, str) for sample_id in ids):
        raise ValueError(f"Label table {path} needs text ids, which name folders")
    repeats = sorted(set(ids[ids.duplicated()]))
    if repeats:
        raise ValueError(f"Label ids must be unique, got {repeats}")
    # An id names its sample's folder below the output directory.
    outside = [
        sample_id
        for sample_id in ids
        if PurePosixPath(sample_id).is_absolute()
        or ".." in PurePosixPath(sample_id).parts
    ]
    if outside:
        raise ValueError(f"Label ids must name folders inside the output: {outside}")
    noncanonical = [
        sample_id for sample_id in ids
        if not PurePosixPath(sample_id).parts
        or PurePosixPath(sample_id).as_posix() != sample_id
    ]
    if noncanonical:
        raise ValueError(f"Label ids must name canonical relative folders: {noncanonical}")

    missing = [
        sample_id
        for sample_id, assets in zip(ids, labels["assets"], strict=True)
        if not isinstance(assets, dict) or "label" not in assets
    ]
    if missing:
        raise ValueError(f"Label rows name no 'label' asset: {missing}")
    return labels
