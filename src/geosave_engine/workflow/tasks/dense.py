"""Prepare one label-aligned dense raster sample."""

from pathlib import Path

from prefect import task
from prefect.cache_policies import NO_CACHE

from geosave_engine.geodata.utils import io
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import RasterRequirement

from .load import load_raster
from .save import write_stack


def _validate_dense_sample(
    path: str | Path,
    requirements: dict[str, RasterRequirement],
) -> None:
    """Validate one completed dense sample."""
    with io.read_stack(path, chunks="auto") as sample:
        if set(sample.gs.groups) != {"label", *requirements}:
            raise ValueError("Existing sample does not match model sources")

        rasters = sample.gs.rasters
        if rasters["label"].gs.timespan is None:
            raise ValueError(f"Existing sample label has no time: {path}")
        _ = sample.gs.anchor
        for name, requirement in requirements.items():
            requirement.validate_raster(rasters[name])


@task(cache_policy=NO_CACHE, persist_result=False)
def _prepare_dense_sample(
    label: str | Path,
    sources: dict[str, SourceConfig],
    requirements: dict[str, RasterRequirement],
    output: str | Path,
) -> str:
    """Prepare one label-aligned dense sample."""
    destination = Path(output)
    if destination.exists():
        _validate_dense_sample(destination, requirements)
        return str(destination)

    with io.read_raster(label) as label_raster:
        anchor = label_raster.gs.anchor
        if anchor.timespan is None:
            raise ValueError(f"Label raster has no time: {label}")

        rasters = {
            name: load_raster(anchor, sources[name], requirement)
            for name, requirement in requirements.items()
        }
        return write_stack({"label": label_raster, **rasters}, destination)
