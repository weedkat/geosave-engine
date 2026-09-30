"""CLI command for ingesting one raster stack."""

from pathlib import Path
from typing import Annotated

from pydantic import TypeAdapter, ValidationError
import typer

from geosave_engine.workflow import flows
from geosave_engine.workflow.configs import AnchorConfig

_anchor_adapter = TypeAdapter(AnchorConfig)


def ingest(
    anchor: Annotated[
        str,
        typer.Option(help="Anchor configuration as JSON.", metavar="JSON"),
    ],
    output: Annotated[Path, typer.Option(help="Destination Zarr path.")],
    spec: Annotated[Path, typer.Option(help="Model specification path.")],
) -> None:
    """Load model rasters on one anchor and write one raster stack."""
    try:
        native_anchor = _anchor_adapter.validate_json(anchor)
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--anchor") from error

    result = flows.ingest(
        anchor=native_anchor.model_dump(
            mode="json",
            exclude_defaults=True,
            exclude_none=True,
        ),
        output=str(output),
        spec=str(spec),
    )
    typer.echo(result)
