"""Run GeoSave workflows locally."""

from pathlib import Path
from typing import Annotated, Any

from pydantic import TypeAdapter, ValidationError
import typer

from geosave_engine.workflow.configs import AnchorConfig, SourceConfig
from geosave_engine.workflow.flows import ingest, prepare_dense_data

workflow_app = typer.Typer(
    help="Run complete GeoSave workflows locally.",
    no_args_is_help=True,
)

_anchor_adapter = TypeAdapter(AnchorConfig)
_sources_adapter = TypeAdapter(dict[str, SourceConfig])


def _parse_json(value: str, adapter: TypeAdapter[Any], option: str) -> Any:
    """Parse and validate one structured command option."""
    try:
        return adapter.validate_json(value)
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint=option) from error


def _primitive(model: Any) -> dict[str, Any]:
    """Return the compact JSON-compatible form of one config model."""
    return model.model_dump(mode="json", exclude_defaults=True, exclude_none=True)


@workflow_app.command("ingest")
def ingest_command(
    anchor: Annotated[
        str,
        typer.Option(help="Anchor configuration as JSON.", metavar="JSON"),
    ],
    output: Annotated[Path, typer.Option(help="Destination Zarr path.")],
    spec: Annotated[Path, typer.Option(help="Model specification path.")],
    sources: Annotated[
        str | None,
        typer.Option(
            help="Source query and loading settings as JSON.",
            metavar="JSON",
        ),
    ] = None,
) -> None:
    """Load model sources on one anchor and write one raster stack."""
    native_anchor = _parse_json(anchor, _anchor_adapter, "--anchor")
    native_sources = (
        None if sources is None else _parse_json(sources, _sources_adapter, "--sources")
    )
    source_values = (
        None
        if native_sources is None
        else {name: _primitive(config) for name, config in native_sources.items()}
    )
    result = ingest(
        anchor=_primitive(native_anchor),
        output=str(output),
        spec=str(spec),
        sources=source_values,
    )
    typer.echo(result)


@workflow_app.command("prepare-dense-data")
def prepare_dense_data_command(
    labels: Annotated[Path, typer.Option(help="Root containing label rasters.")],
    output: Annotated[Path, typer.Option(help="Prepared data directory.")],
    spec: Annotated[Path, typer.Option(help="Model specification path.")],
    pattern: Annotated[str, typer.Option(help="Recursive label glob.")] = "**/*.tif",
    sources: Annotated[
        str | None,
        typer.Option(
            help="Source query and loading settings as JSON.",
            metavar="JSON",
        ),
    ] = None,
    max_concurrency: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum simultaneous complete sample ingestions.",
        ),
    ] = 1,
) -> None:
    """Prepare label-aligned samples and publish their manifest."""
    native_sources = (
        None if sources is None else _parse_json(sources, _sources_adapter, "--sources")
    )
    source_values = (
        None
        if native_sources is None
        else {name: _primitive(config) for name, config in native_sources.items()}
    )
    result = prepare_dense_data(
        labels=str(labels),
        output=str(output),
        spec=str(spec),
        pattern=pattern,
        sources=source_values,
        max_concurrency=max_concurrency,
    )
    typer.echo(result)
