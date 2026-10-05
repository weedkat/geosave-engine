"""Prepare dense label rasters and matching imagery for training."""

from pathlib import Path
from typing import Literal

from pydantic import JsonValue, TypeAdapter
import typer
from dotenv import load_dotenv

import geosave_engine as gs
from geosave_engine.workflow.flows import prepare_dense_data

_write_options_adapter = TypeAdapter(dict[str, JsonValue])


def main(
    labels: Path = Path("data/labels"),
    output: Path = Path("data/prepared"),
    spec: Path = Path("configs/model_spec.yaml"),
    pattern: str = "**/*.tif",
    format: Literal["geotiff", "zarr"] = "geotiff",
    write_options: str = "{}",
) -> None:
    """Prepare labels and imagery, then print the GeoParquet manifest path.

    Args:
        labels: Label table as GeoParquet, or a local root of label rasters.
        output: Local directory for prepared samples and the manifest.
        spec: Model specification defining required imagery.
        pattern: Recursive label glob, for a root of label rasters.
        format: Persisted sample representation.
        write_options: Native writer options as JSON.
    """
    load_dotenv()
    gs.configure_gdal(gdal_disable_readdir_on_open=True, gdal_http_max_retry=3)
    manifest = prepare_dense_data(
        labels=str(labels),
        output=str(output),
        spec=str(spec),
        pattern=pattern,
        format=format,
        write_options=_write_options_adapter.validate_json(write_options),
    )
    typer.echo(manifest)


if __name__ == "__main__":
    typer.run(main)
