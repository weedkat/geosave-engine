"""Prepare bounded, label-aligned dense training samples."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.futures import as_completed
from pydantic import JsonValue, PositiveInt, TypeAdapter

from geosave_engine.geodata.utils import io
from geosave_engine.model_spec import ModelSpec, RasterRequirement

from .manifest import (
    find_labels as _find_labels,
    read_sample_metadata,
    sample_path as _sample_path,
    write_manifest,
)
from .sample import SampleFormat, open_sample, write_sample


def _validate_dense_sample(
    path: str | Path,
    requirements: dict[str, RasterRequirement],
    *,
    format: SampleFormat = "geotiff",
) -> None:
    """Validate one completed dense sample."""
    with open_sample(path, format=format) as sample:
        if set(sample.gs.groups) != {"label", *requirements}:
            raise ValueError("Existing sample does not match model rasters")

        rasters = sample.gs.rasters
        if rasters["label"].gs.timespan is None:
            raise ValueError(f"Existing sample label has no time: {path}")
        _ = sample.gs.anchor
        for name, requirement in requirements.items():
            raster = rasters[name]
            if (
                format == "geotiff"
                and requirement.dims is not None
                and "time" in requirement.dims
                and "time" in raster.coords
                and "time" not in raster.dims
            ):
                raster = raster.expand_dims("time").transpose(*requirement.dims)
            requirement.validate_raster(raster)


@task(cache_policy=NO_CACHE, persist_result=False)
def prepare_dense_sample(
    label: str | Path,
    model: ModelSpec,
    output: str | Path,
    *,
    format: SampleFormat = "geotiff",
    write_options: Mapping[str, JsonValue] | None = None,
) -> str:
    """Prepare one label-aligned dense sample."""
    model = ModelSpec.model_validate(model.model_dump())
    destination = Path(output)
    if destination.exists():
        _validate_dense_sample(destination, model.rasters, format=format)
        return str(destination)

    with io.read_raster(label) as label_raster:
        anchor = label_raster.gs.anchor
        if anchor.timespan is None:
            raise ValueError(f"Label raster has no time: {label}")

        rasters = model.load_rasters(anchor)
        return write_sample(
            {"label": label_raster, **rasters},
            destination,
            format=format,
            write_options=write_options,
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
    missing = [
        name
        for name, requirement in model.rasters.items()
        if requirement.stac is None
    ]
    if missing:
        raise ValueError(f"STAC recipes required for rasters: {missing}")

    destination = Path(output)
    discovered = _find_labels(Path(labels), pattern)
    properties = read_sample_metadata(metadata, discovered)
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
        ordered,
        destination / "manifest.parquet",
        format=format,
        metadata=properties,
    )
