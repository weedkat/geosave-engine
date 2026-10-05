"""Prepare label-aligned dense training samples."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
from prefect import task
from prefect.cache_policies import NO_CACHE
from pydantic import JsonValue

from geosave_engine.geodata import GeoDataFrame, GeoVector, read_stack
from geosave_engine.geodata import io
from geosave_engine.model.spec import ModelSpec, RasterRequirement

from .sample import SampleFormat, write_sample


def validate_row(row: pd.Series, spec: ModelSpec) -> None:
    """Refuse a sample row that names other layers than the spec asks for.

    Args:
        row: One manifest row.
        spec: Model spec the sample must serve.

    Raises:
        ValueError: The row's assets are not the label and the spec's rasters,
            or a raster lacks a variable the spec requires.
    """
    assets = cast("dict[str, dict[str, Any]]", row["assets"])
    if set(assets) != {"label", *spec.rasters}:
        raise ValueError(
            f"Existing sample does not match model rasters: {row['id']!r} holds "
            f"{sorted(assets)}"
        )
    for name, requirement in spec.rasters.items():
        bands = {band["name"] for band in assets[name].get("bands") or ()}
        absent = sorted(set(requirement.variables or ()) - bands)
        if absent:
            raise ValueError(
                f"Existing sample does not match model rasters: {row['id']!r} "
                f"holds no {absent} in {name!r}"
            )


def _validate_dense_sample(
    source: str | Path, requirements: dict[str, RasterRequirement]
) -> None:
    """Validate one completed dense sample."""
    with read_stack(source) as sample:
        if set(sample.gs.groups) != {"label", *requirements}:
            raise ValueError("Existing sample does not match model rasters")

        rasters = sample.gs.rasters
        if rasters["label"].gs.timespan is None:
            raise ValueError(f"Existing sample label has no time: {source}")
        if sample.gs.geobox is None:
            raise ValueError(f"Existing sample rasters do not share a grid: {source}")
        for name, requirement in requirements.items():
            requirement.select_raster(rasters[name])


@task(cache_policy=NO_CACHE, persist_result=False)
def prepare_dense_sample(
    label_path: str | Path,
    spec: ModelSpec,
    output: str | Path,
    *,
    sample_id: str | None = None,
    properties: Mapping[str, object] | None = None,
    format: SampleFormat = "geotiff",
    write_options: Mapping[str, JsonValue] | None = None,
) -> GeoDataFrame:
    """Prepare one label-aligned dense sample and describe it as a row.

    Args:
        label_path: Label raster, which sets the sample's grid and time.
        spec: Model spec whose rasters are loaded onto the label.
        output: Sample directory. An existing one is validated and kept.
        sample_id: Row identifier. None uses the directory's name.
        properties: Caller columns carried onto the row.
        format: Persisted representation of each sample raster.
        write_options: Serializable options for the native raster writer.

    Returns:
        One-row STAC table naming the sample's layers as assets.

    Raises:
        ValueError: The label has no time, or an existing sample does not
            match the spec's rasters.
    """
    destination = Path(output)
    if destination.exists():
        _validate_dense_sample(destination, spec.rasters)
    else:
        with io.read_raster(label_path) as label:
            anchor = label.gs.anchor
            if anchor.timespan is None:
                raise ValueError(f"Label raster has no time: {label_path}")

            write_sample(
                {"label": label, **spec.load_rasters(anchor)},
                destination,
                format=format,
                write_options=write_options,
            )
    return GeoVector.from_xarray(
        read_stack(destination),
        id=destination.name if sample_id is None else sample_id,
        properties=properties,
    )
