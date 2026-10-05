"""What an xarray object's attrs mean, as typed models.

Flat attrs are a namespace odc, rioxarray and user code all write to. This
package does not own that namespace; it decodes, encodes, and says what a key
means. Whoever changes the data owns the attrs describing it.
"""

from .header import AttrsHeader, DroppedAttr
from .model import (
    MUST_AGREE,
    AttrsModel,
    FlatAttrs,
    attrs_equal,
    common_attrs,
    parse_field_value,
)
from .models import (
    ACDD,
    CELL_METHODS,
    MODELS,
    CFCoordinate,
    CFVariable,
    GDALVariable,
    GeoTIFFTags,
    Legend,
    Nodata,
    Packing,
    Scope,
    StacItem,
    StacMetadata,
    StackedAttrs,
    TimeSpec,
    ZarrOrder,
    model_scope,
    resolve_model,
)
from .namespace import AttrsNamespace
from .xarray import AttrsEdit, XarrayObject, create_header, flag_variables, merge, rebase

__all__ = [
    "ACDD",
    "CELL_METHODS",
    "MUST_AGREE",
    "MODELS",
    "Scope",
    "AttrsEdit",
    "AttrsHeader",
    "AttrsNamespace",
    "AttrsModel",
    "CFCoordinate",
    "CFVariable",
    "DroppedAttr",
    "FlatAttrs",
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
    "common_attrs",
    "XarrayObject",
    "create_header",
    "rebase",
    "model_scope",
    "resolve_model",
    "parse_field_value",
]
