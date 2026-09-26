"""Prepare dense raster samples and their spatial manifest."""

from pathlib import Path

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from pydantic import JsonValue

from geosave_engine.geodata.utils import io
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement
from geosave_engine.workflow.tasks.catalog import write_manifest
from geosave_engine.workflow.tasks.load import load_raster, source_concurrency
from geosave_engine.workflow.tasks.save import write_stack

__all__ = ["prepare", "validate_sample"]


def validate_sample(
    path: str | Path, requirements: dict[str, RasterRequirement]
) -> None:
    """Validate a dense sample against its model source requirements.

    Args:
        path: Completed sample Zarr path.
        requirements: Model source requirements keyed by sample group.

    Raises:
        ValueError: If groups, label time, grid metadata, or source rasters are
            invalid.
    """
    with io.read_stack(path, chunks="auto") as sample:
        if set(sample.gs.groups) != {"label", *requirements}:
            raise ValueError("Existing sample does not match model sources")

        rasters = sample.gs.rasters
        if rasters["label"].gs.timespan is None:
            raise ValueError(f"Existing sample label has no time: {path}")
        _ = sample.gs.anchor
        for name, requirement in requirements.items():
            requirement.validate_raster(rasters[name])


def _discover_labels(root: Path, pattern: str) -> dict[str, Path]:
    """Return sorted label paths keyed by their relative sample IDs."""
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No labels found for {pattern!r} under {root}")

    labels = {path.relative_to(root).as_posix(): path for path in paths}
    outputs = [Path(name).with_suffix(".zarr") for name in labels]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Labels must map to unique sample paths")
    return labels


@task(cache_policy=NO_CACHE, persist_result=False)
def _prepare_sample(
    label: str | Path,
    sources: dict[str, SourceConfig],
    requirements: dict[str, RasterRequirement],
    output: str | Path,
) -> str:
    """Prepare one label-aligned dense sample."""
    destination = Path(output)
    if destination.exists():
        validate_sample(destination, requirements)
        return str(destination)

    with io.read_raster(label) as label_raster:
        anchor = label_raster.gs.anchor
        if anchor.timespan is None:
            raise ValueError(f"Label raster has no time: {label}")

        with source_concurrency(sources.values()):
            rasters = {
                name: load_raster(anchor, sources[name], requirement)
                for name, requirement in requirements.items()
            }
            return write_stack({"label": label_raster, **rasters}, destination)


@flow(name="dense-prepare", persist_result=False)
def prepare(
    labels: str,
    sources: dict[str, dict[str, JsonValue]],
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
) -> str:
    """Prepare dense training samples and publish their spatial manifest.

    Args:
        labels: Local root containing label rasters.
        sources: Runtime query and loading settings keyed by model source.
        output: Directory for sample Zarrs and the GeoParquet manifest.
        spec: Model specification defining required raster sources.
        pattern: Recursive label glob relative to ``labels``.

    Returns:
        Path to the completed GeoParquet manifest.

    Raises:
        ValueError: If source bindings, label discovery, or dense sample
            requirements are invalid.
    """
    configs = {
        name: SourceConfig.model_validate(config) for name, config in sources.items()
    }
    requirements = ModelSpec.load(spec).sources
    if configs.keys() != requirements.keys():
        raise ValueError("Source bindings must match model sources")
    if "label" in requirements:
        raise ValueError("Model source name 'label' is reserved")

    destination = Path(output)
    discovered = _discover_labels(Path(labels), pattern)
    pending = {
        sample_id: _prepare_sample.submit(
            label,
            configs,
            requirements,
            destination / "samples" / Path(sample_id).with_suffix(".zarr"),
        )
        for sample_id, label in discovered.items()
    }
    completed = {
        sample_id: future.result() for sample_id, future in pending.items()
    }
    return write_manifest.submit(
        completed, destination / "manifest.parquet"
    ).result()
