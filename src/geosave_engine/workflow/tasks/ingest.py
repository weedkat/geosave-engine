"""Prepare one label raster and its matching imagery."""

from pathlib import Path

from prefect import task
from prefect.cache_policies import NO_CACHE

from geosave_engine.geodata.utils import io
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import RasterRequirement

from .load import RasterLoader
from .save import write_stack


def _validate_sample(
    path: Path, requirements: dict[str, RasterRequirement]
) -> None:
    """Validate a completed sample without loading its pixel values."""
    with io.read_stack(path, chunks="auto") as sample:
        expected = {"label", *requirements}
        actual = set(sample.gs.groups)
        if actual != expected:
            raise ValueError(
                f"Existing sample groups {sorted(actual)} do not match "
                f"{sorted(expected)}"
            )

        rasters = sample.gs.rasters
        if rasters["label"].gs.timespan is None:
            raise ValueError(f"Existing sample label has no time: {path}")
        sample.gs.anchor
        for name, requirement in requirements.items():
            try:
                requirement.select_raster(rasters[name])
            except Exception as error:
                error.add_note(f"While validating source {name!r} in {path}")
                raise


@task(cache_policy=NO_CACHE, persist_result=False)
def ingest_sample(
    label: str | Path,
    sources: dict[str, SourceConfig],
    requirements: dict[str, RasterRequirement],
    output: str | Path,
) -> str:
    """Write one label raster and its matching imagery as a completed stack."""
    destination = Path(output)
    if destination.exists():
        _validate_sample(destination, requirements)
        return str(destination)

    with io.read_raster(label) as label_raster:
        anchor = label_raster.gs.anchor
        if anchor.timespan is None:
            raise ValueError(f"Label raster has no time: {label}")

        rasters = {}
        for name, requirement in requirements.items():
            try:
                rasters[name] = RasterLoader(requirement).load(sources[name], anchor)
            except Exception as error:
                error.add_note(f"While loading source {name!r}")
                raise
        return write_stack({"label": label_raster, **rasters}, destination)
