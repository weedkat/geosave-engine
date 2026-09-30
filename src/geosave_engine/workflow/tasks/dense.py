"""Prepare label-aligned dense training samples."""

from collections.abc import Mapping
from pathlib import Path

from prefect import task
from prefect.cache_policies import NO_CACHE
from pydantic import JsonValue

from geosave_engine.geodata.utils import io
from geosave_engine.model_spec import ModelSpec, RasterRequirement

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
