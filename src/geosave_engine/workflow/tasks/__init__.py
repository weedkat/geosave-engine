"""Short Prefect tasks used by deployable workflow flows."""

from .load import load_raster
from .save import save_stack

__all__ = ["load_raster", "save_stack"]
