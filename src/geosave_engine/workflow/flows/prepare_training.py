"""Prepare raster training samples from a local label tree."""

from pathlib import Path

from prefect import flow
from pydantic import JsonValue

from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import ModelSpec
from geosave_engine.workflow.tasks import prepare_sample, save_catalog


def _labels(root: Path, pattern: str) -> dict[str, Path]:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No labels found for {pattern!r} under {root}")

    labels = {path.relative_to(root).as_posix(): path for path in paths}
    outputs = [Path(name).with_suffix(".zarr") for name in labels]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Labels must map to unique sample paths")
    return labels


@flow(name="prepare-training", persist_result=False)
def prepare_training(
    labels: str,
    sources: dict[str, dict[str, JsonValue]],
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
) -> str:
    """Prepare label-aligned samples and publish their GeoVector manifest."""
    configs = {
        name: SourceConfig.model_validate(config) for name, config in sources.items()
    }
    requirements = ModelSpec.load(spec).sources
    if configs.keys() != requirements.keys():
        raise ValueError("Source bindings must match model sources")
    if "label" in requirements:
        raise ValueError("Model source name 'label' is reserved")

    destination = Path(output)
    discovered = _labels(Path(labels), pattern)
    pending = {
        sample_id: prepare_sample.submit(
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
    return save_catalog.submit(
        completed, destination / "manifest.parquet"
    ).result()
