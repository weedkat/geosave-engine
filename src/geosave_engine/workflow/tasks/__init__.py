"""Short Prefect tasks used by deployable workflow flows."""

from .call import invoke_call
from .ingest import ingest_sample
from .load import load_raster
from .save import save_stack

__all__ = ["ingest_sample", "invoke_call", "load_raster", "save_stack"]
