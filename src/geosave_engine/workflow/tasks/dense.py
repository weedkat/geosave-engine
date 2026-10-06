"""Public dense sample catalog operations awaiting the native Item contract."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from prefect import task
from prefect.cache_policies import NO_CACHE
from pydantic import JsonValue

from geosave_engine.geodata import GeoDataFrame
from geosave_engine.model.spec import ModelSpec
from .sample import SampleFormat


def validate_row(row: pd.Series, spec: ModelSpec) -> None:
    """Validate a sample against the future native Item asset mapping.

    Args:
        row: Selected native catalog row.
        spec: Model requirements.

    Raises:
        NotImplementedError: Sample asset validation is not implemented.
    """
    raise NotImplementedError("Dense sample catalog construction is not implemented")


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
    """Prepare a cataloged dense sample after the native Item contract is ready.

    Args:
        label_path: Source label raster.
        spec: Model specification.
        output: Prepared sample destination.
        sample_id: Item identity.
        properties: Item annotations.
        format: Raster storage format.
        write_options: Native pixel writer options.

    Returns:
        Native GeoDataFrame describing the completed sample.

    Raises:
        NotImplementedError: Sample catalog construction is not implemented.
    """
    raise NotImplementedError("Dense sample catalog construction is not implemented")
