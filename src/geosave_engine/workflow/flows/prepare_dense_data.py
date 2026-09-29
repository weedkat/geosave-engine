"""Prepare bounded, label-aligned dense training samples."""

from pathlib import Path
from typing import Literal

from prefect import flow
from prefect.futures import as_completed
from pydantic import JsonValue, PositiveInt, TypeAdapter

from geosave_engine.model_spec import ModelSpec
from geosave_engine.workflow.tasks import prepare_dense_sample, write_manifest


def _find_labels(root: Path, pattern: str) -> dict[str, Path]:
    """Return sorted label paths keyed by their relative sample IDs."""
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No labels found for {pattern!r} under {root}")

    labels = {
        path.relative_to(root).with_suffix("").as_posix(): path for path in paths
    }
    if len(labels) != len(paths):
        raise ValueError("Labels must map to unique sample paths")
    return labels


def _sample_path(
    root: Path, sample_id: str, format: Literal["geotiff", "zarr"]
) -> Path:
    """Return the format-specific path for one suffix-free sample ID."""
    path = root / sample_id
    if format == "zarr":
        return path.parent / f"{path.name}.zarr"
    return path


@flow(name="prepare-dense-data", persist_result=False)
def prepare_dense_data(
    labels: str,
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
    max_concurrency: PositiveInt = 1,
    format: Literal["geotiff", "zarr"] = "geotiff",
    write_options: dict[str, JsonValue] | None = None,
) -> str:
    """Prepare dense training samples and publish their spatial manifest.

    Args:
        labels: Local root containing label rasters.
        output: Directory for prepared samples and the GeoParquet manifest.
        spec: Model specification defining required rasters and STAC recipes.
        pattern: Recursive label glob relative to ``labels``.
        max_concurrency: Maximum number of active sample ingestions. Defaults
            to one to protect raster reads and writes.
        format: Persisted sample representation.
        write_options: Serializable options for the native raster writer.

    Returns:
        Path to the completed GeoParquet manifest.

    Raises:
        ValueError: If recipes, label discovery, concurrency, or sample
            requirements are invalid.
    """
    limit = TypeAdapter(PositiveInt).validate_python(max_concurrency)
    model = ModelSpec.load(spec)
    if "label" in model.rasters:
        raise ValueError("Model raster name 'label' is reserved")
    missing = [
        name
        for name, requirement in model.rasters.items()
        if requirement.stac is None
    ]
    if missing:
        raise ValueError(f"STAC recipes required for rasters: {missing}")

    destination = Path(output)
    discovered = _find_labels(Path(labels), pattern)
    remaining = iter(discovered.items())
    pending = {}
    completed = {}

    for sample_id, label in remaining:
        future = prepare_dense_sample.submit(
            label,
            model,
            _sample_path(destination, sample_id, format),
            format=format,
            write_options=write_options,
        )
        pending[future] = sample_id
        if len(pending) == limit:
            break

    while pending:
        future = next(as_completed(list(pending)))
        sample_id = pending.pop(future)
        completed[sample_id] = future.result()

        try:
            next_sample_id, label = next(remaining)
        except StopIteration:
            continue
        next_future = prepare_dense_sample.submit(
            label,
            model,
            _sample_path(destination, next_sample_id, format),
            format=format,
            write_options=write_options,
        )
        pending[next_future] = next_sample_id

    ordered = {sample_id: completed[sample_id] for sample_id in discovered}
    return write_manifest(
        ordered, destination / "manifest.parquet", format=format
    )
