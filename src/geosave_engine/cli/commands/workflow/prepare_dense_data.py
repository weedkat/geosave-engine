"""CLI command for preparing dense training data."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError
import typer

from geosave_engine.workflow import training_data

_write_options_adapter = TypeAdapter(dict[str, JsonValue])


def prepare_dense_data(
    labels: Annotated[Path, typer.Option(help="Root containing label rasters.")],
    output: Annotated[Path, typer.Option(help="Prepared data directory.")],
    spec: Annotated[Path, typer.Option(help="Model specification path.")],
    pattern: Annotated[str, typer.Option(help="Recursive label glob.")] = "**/*.tif",
    max_concurrency: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum simultaneous complete sample ingestions.",
        ),
    ] = 1,
    format: Annotated[
        Literal["geotiff", "zarr"],
        typer.Option(help="Prepared sample format."),
    ] = "geotiff",
    write_options: Annotated[
        str,
        typer.Option(help="Native writer options as JSON.", metavar="JSON"),
    ] = "{}",
    metadata: Annotated[
        Path | None,
        typer.Option(help="CSV, TSV, Parquet, or XLSX sample metadata."),
    ] = None,
) -> None:
    """Prepare label-aligned samples and publish their manifest."""
    try:
        native_write_options = _write_options_adapter.validate_json(write_options)
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--write-options") from error

    result = training_data.prepare_dense_data(
        labels=str(labels),
        output=str(output),
        spec=str(spec),
        pattern=pattern,
        max_concurrency=max_concurrency,
        format=format,
        write_options=native_write_options,
        metadata=None if metadata is None else str(metadata),
    )
    typer.echo(result)
