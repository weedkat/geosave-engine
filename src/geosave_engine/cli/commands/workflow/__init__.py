"""Run GeoSave workflows locally."""

import typer

from .ingest import ingest
from .prepare_dense_data import prepare_dense_data

workflow = typer.Typer(
    help="Run complete GeoSave workflows locally.",
    no_args_is_help=True,
)
workflow.command()(ingest)
workflow.command()(prepare_dense_data)
