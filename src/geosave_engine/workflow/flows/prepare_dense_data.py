"""Prepare bounded, label-aligned dense training datasets."""

from pathlib import Path
from typing import Literal

from prefect import flow
from prefect.futures import as_completed
from pydantic import JsonValue, PositiveInt, TypeAdapter

from geosave_engine.model_spec import ModelSpec
from geosave_engine.workflow.tasks import prepare_dense_sample
from geosave_engine.workflow.tasks.manifest import (
    find_labels,
    read_sample_metadata,
    sample_path,
    write_manifest,
)


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
    metadata: str | None = None,
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
        metadata: Optional CSV, TSV, Parquet, or XLSX sample metadata table.

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
    model.require_recipes()

    destination = Path(output)
    discovered = find_labels(Path(labels), pattern)
    properties = read_sample_metadata(metadata, discovered)
    remaining = iter(discovered.items())
    pending = {}
    completed = {}

    for sample_id, label in remaining:
        future = prepare_dense_sample.submit(
            label,
            model,
            sample_path(destination, sample_id, format),
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
            sample_path(destination, next_sample_id, format),
            format=format,
            write_options=write_options,
        )
        pending[next_future] = next_sample_id

    ordered = {sample_id: completed[sample_id] for sample_id in discovered}
    return write_manifest(
        ordered,
        destination / "manifest.parquet",
        format=format,
        metadata=properties,
    )
