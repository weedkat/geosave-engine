"""Ingest one model-ready raster sample on an explicit anchor."""

from pathlib import Path
from typing import Literal

from prefect import flow
from pydantic import JsonValue

from geosave_engine.geodata import GeoDataFrame, GeoVector, read_stack, read_vector
from geosave_engine.model.spec import ModelSpec
from geosave_engine.workflow.configs import AnchorConfig
from geosave_engine.workflow.tasks.sample import write_sample


@flow(name="ingest", persist_result=False)
def ingest(
    anchor: AnchorConfig,
    *,
    output: str,
    spec: str,
    format: Literal["geotiff", "zarr"] = "zarr",
    write_options: dict[str, JsonValue] | None = None,
    catalog: str | None = None,
) -> str:
    """Load model rasters on one anchor and write them as one sample.

    Args:
        anchor: Anchor configuration defining grid and time.
        output: New directory, which takes one raster per model raster.
        spec: Model specification defining required rasters and STAC recipes.
        format: Persisted representation of each raster. GeoTIFF holds one
            instant; Zarr holds a time series.
        write_options: Serializable options for the native raster writer.
        catalog: GeoParquet STAC table recording what was written. The
            sample's row, named after the output directory, is added to it
            or replaces the row already carrying that name. None records
            nothing.

    Returns:
        Path to the completed sample directory.

    Raises:
        FileExistsError: If the output directory already exists.
        ValueError: If a raster declares no `stac` block or fails validation,
            or the catalog is not a GeoParquet STAC table in
            longitude/latitude.
    """
    # A catalog that cannot take the row is refused before anything is loaded.
    table = None if catalog is None else _read_catalog(Path(catalog))

    rasters = ModelSpec.load(spec).load_rasters(anchor.open())
    folder = write_sample(rasters, output, format=format, write_options=write_options)
    if catalog is not None:
        row = GeoVector.from_xarray(read_stack(folder), id=Path(folder).name)
        rows = row if table is None else table.gs.upsert(row, on="id")
        Path(catalog).parent.mkdir(parents=True, exist_ok=True)
        rows.gs.to_geoparquet(catalog, overwrite=True)
    return folder


def _read_catalog(path: Path) -> GeoDataFrame | None:
    """Read the catalog a row will be added to, if it exists yet."""
    if path.suffix.lower() not in (".parquet", ".geoparquet"):
        raise ValueError(f"The catalog {path} must be a GeoParquet table")
    if not path.exists():
        return None
    rows = read_vector(path)
    if "id" not in rows:
        raise ValueError(f"The catalog {path} has no 'id' column naming its rows")
    if rows.crs != "EPSG:4326":
        raise ValueError(
            f"The catalog {path} is in {rows.crs}; a STAC table states footprints "
            f"in EPSG:4326"
        )
    return rows
