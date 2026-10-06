"""CLI command for ingesting one raster sample."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError
import typer

from geosave_engine.workflow import flows
from geosave_engine.workflow.configs import AnchorConfig

_anchor_adapter = TypeAdapter(AnchorConfig)
_write_options_adapter = TypeAdapter(dict[str, JsonValue])


def ingest(
    anchor: Annotated[
        str,
        typer.Option(help="Anchor configuration as JSON.", metavar="JSON"),
    ],
    output: Annotated[Path, typer.Option(help="New sample directory.")],
    spec: Annotated[Path, typer.Option(help="Model specification path.")],
    format: Annotated[
        Literal["geotiff", "zarr"],
        typer.Option(help="Format of each written raster."),
    ] = "zarr",
    write_options: Annotated[
        str,
        typer.Option(help="Native writer options as JSON.", metavar="JSON"),
    ] = "{}",
) -> None:
    """Load model rasters on one anchor and write them as one sample."""
    try:
        config = _anchor_adapter.validate_json(anchor)
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--anchor") from error
    try:
        native_write_options = _write_options_adapter.validate_json(write_options)
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--write-options") from error
    typer.echo(
        flows.ingest(
            anchor=config,
            output=str(output),
            spec=str(spec),
            format=format,
            write_options=native_write_options,
        )
    )
