"""Public dense preparation flow awaiting the native Item contract."""

from typing import Literal

from prefect import flow
from pydantic import JsonValue, PositiveInt


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
    """Prepare a dense dataset after native Item construction is implemented.

    Args:
        labels: Label raster directory or catalog.
        output: Prepared dataset directory.
        spec: Model specification path.
        pattern: Label file glob.
        max_concurrency: Maximum simultaneous sample jobs.
        format: Raster storage format.
        write_options: Native pixel writer options.

    Returns:
        Completed sample catalog path.

    Raises:
        NotImplementedError: Dense catalog preparation is not implemented.
    """
    raise NotImplementedError("Dense sample catalog construction is not implemented")
