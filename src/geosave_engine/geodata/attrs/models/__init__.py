"""Every attrs model GeoSave reads and writes, and where each one lives."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from types import MappingProxyType
from typing import Literal

from geosave_engine.geodata.attrs.model import AttrsModel

from .acdd import ACDD
from .cf import CELL_METHODS, CFCoordinate, CFVariable
from .gdal import GDALVariable
from .geotiff import GeoTIFFTags
from .legend import Legend
from .nodata import Nodata
from .packing import Packing
from .stac import StacItem, StacMetadata
from .stacked import StackedAttrs
from .timespec import TimeSpec
from .zarr import ZarrOrder

type Scope = Literal["dataset", "variable", "coordinate"]

# Where each model lives, after the attribute usage in CF Appendix A.
MODELS: Mapping[Scope, tuple[type[AttrsModel], ...]] = MappingProxyType(
    {
        "dataset": (ACDD, GeoTIFFTags, StacMetadata, ZarrOrder),
        "variable": (CFVariable, GDALVariable, Legend, Nodata, Packing),
        "coordinate": (CFCoordinate, TimeSpec, StackedAttrs),
    }
)


def resolve_model(model: type[AttrsModel] | str) -> type[AttrsModel]:
    """Find a GeoSave attrs model by class or `NAME`.

    Args:
        model: Model class or its `NAME`.

    Returns:
        The model class listed in `MODELS`.

    Raises:
        KeyError: No model has this `NAME`.
        TypeError: The class is not listed in `MODELS`.

    Examples:
        >>> resolve_model("acdd")
        <class '...ACDD'>
    """
    for models in MODELS.values():
        for candidate in models:
            if candidate is model or candidate.NAME == model:
                return candidate
    if isinstance(model, str):
        raise KeyError(f"no attrs model is named {model!r}")
    raise TypeError(f"{model!r} is not a GeoSave attrs model")


def model_scope(model: type[AttrsModel] | str) -> Scope:
    """Name the scope a model belongs to.

    Args:
        model: Model class or its `NAME`.

    Returns:
        `"dataset"`, `"variable"`, or `"coordinate"`.

    Raises:
        KeyError: No model has this `NAME`.
        TypeError: The class is not listed in `MODELS`.
    """
    model = resolve_model(model)
    return next(scope for scope, models in MODELS.items() if model in models)


@cache
def scope_keys(scope: Scope) -> frozenset[str]:
    """Return every attr key a model in `scope` writes."""
    return frozenset(key for model in MODELS[scope] for key in model.attr_keys())


__all__ = [
    "ACDD",
    "CELL_METHODS",
    "MODELS",
    "CFCoordinate",
    "CFVariable",
    "GDALVariable",
    "GeoTIFFTags",
    "Legend",
    "Nodata",
    "Packing",
    "Scope",
    "StacItem",
    "StacMetadata",
    "StackedAttrs",
    "TimeSpec",
    "ZarrOrder",
    "model_scope",
    "resolve_model",
    "scope_keys",
]
