"""Short Prefect tasks used by deployable workflow flows."""

from .call import invoke_call
from .catalog import save_catalog
from .ingest import ingest_sample
from .load import load_raster
from .save import save_stack

__all__ = [
    "ingest_sample",
    "invoke_call",
    "load_raster",
    "save_catalog",
    "save_stack",
]
