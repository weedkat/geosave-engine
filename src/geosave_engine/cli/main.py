import typer

from geosave_engine.cli.commands.create import create
from geosave_engine.cli.commands.infra import infra_app
from geosave_engine.cli.commands.upload import upload

app = typer.Typer(help="GeoSave Engine CLI", no_args_is_help=True)
app.command()(create)
app.command()(upload)

app.add_typer(infra_app, name="infra")
