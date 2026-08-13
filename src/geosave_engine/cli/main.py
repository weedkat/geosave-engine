import typer
from dotenv import load_dotenv

from .commands.create import create
from .commands.make import make

app = typer.Typer(
    help="GeoSave Engine CLI", 
    no_args_is_help=True, 
    add_completion=True,
)
app.command()(create)
app.command()(make)


@app.callback()
def global_callback() -> None:
    """Load a .env from the current directory (or a parent), if any, before any command runs."""
    load_dotenv()

if __name__ == "__main__":
    app()