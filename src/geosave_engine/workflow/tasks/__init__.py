"""Short Prefect tasks used by deployable workflow flows."""

from .call import invoke_call
from .catalog import save_catalog
from .ingest import ingest_sample

__all__ = [
    "ingest_sample",
    "invoke_call",
    "save_catalog",
]
