"""What an xarray object's attrs mean, as typed models.

Flat attrs are a namespace odc, rioxarray and user code all write to. This
package does not own that namespace; it decodes, encodes, and says what a key
means. Whoever changes the data owns the attrs describing it.
"""

from .header import AttrsHeader, DroppedAttr
from .headers.xarray import create_header
from .namespace import AttrsNamespace
from .model import (
    REGISTERED_MODELS,
    AttrsModel,
    attrs_equal,
    parse_field_value,
    resolve_model,
)
from .models import (
    ACDD,
    CELL_METHODS,
    CFCoordinate,
    CFVariable,
    GDALVariable,
    GeoTIFFTags,
    Legend,
    Nodata,
    Packing,
    StacItem,
    StacMetadata,
    StackedAttrs,
    TimeSpec,
    ZarrOrder,
)
from .xarray import flag_variables, merge, rebase

__all__ = [
    "ACDD",
    "CELL_METHODS",
    "AttrsHeader",
    "AttrsNamespace",
    "AttrsModel",
    "CFCoordinate",
    "CFVariable",
    "DroppedAttr",
    "GDALVariable",
    "GeoTIFFTags",
    "Legend",
    "Nodata",
    "Packing",
    "StacItem",
    "StacMetadata",
    "StackedAttrs",
    "TimeSpec",
    "ZarrOrder",
    "flag_variables",
    "merge",
    "attrs_equal",
    "create_header",
    "rebase",
    "REGISTERED_MODELS",
    "resolve_model",
    "parse_field_value",
]
