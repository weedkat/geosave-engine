"""Prepare label rasters and matching imagery as a training dataset."""

from pathlib import Path

import typer
from dotenv import load_dotenv

import geosave_engine as gs
from geosave_engine.workflow.flows import ingest

SOURCES = {"sentinel_2_l2a": {"query": {}, "load": {}}}


def main(
    labels: Path = Path("data/labels"),
    output: Path = Path("data/prepared"),
    spec: Path = Path("model_spec.yaml"),
    pattern: str = "**/*.tif",
) -> None:
    """Prepare labels and imagery, then print the GeoParquet manifest path.

    Args:
        labels: Local root containing label rasters.
        output: Local directory for sample Zarrs and the manifest.
        spec: Model specification defining required imagery.
        pattern: Recursive label glob relative to `labels`.
    """
    load_dotenv()
    gs.configure_gdal(gdal_disable_readdir_on_open=True, gdal_http_max_retry=3)
    manifest = ingest(
        labels=str(labels),
        sources=SOURCES,
        output=str(output),
        spec=str(spec),
        pattern=pattern,
    )
    typer.echo(manifest)


if __name__ == "__main__":
    typer.run(main)
