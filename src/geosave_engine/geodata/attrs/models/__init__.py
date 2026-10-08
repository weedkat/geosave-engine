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
from .spectral import Spectral
from .stac import StacItem, StacMetadata
from .stacked import StackedAttrs
from .timespec import TimeSpec

type Scope = Literal["dataset", "variable", "coordinate"]

# Where each model lives, after the attribute usage in CF Appendix A.
MODELS: Mapping[Scope, Mapping[str, type[AttrsModel]]] = MappingProxyType(
    {
        "dataset": MappingProxyType(
            {
                "acdd": ACDD,
                "geotiff_tags": GeoTIFFTags,
                "stac_metadata": StacMetadata,
            }
        ),
        "variable": MappingProxyType(
            {
                "cf_variable": CFVariable,
                "gdal_variable": GDALVariable,
                "legend": Legend,
                "nodata": Nodata,
                "packing": Packing,
                "spectral": Spectral,
            }
        ),
        "coordinate": MappingProxyType(
            {
                "cf_coordinate": CFCoordinate,
                "time_spec": TimeSpec,
                "stacked_attrs": StackedAttrs,
            }
        ),
    }
)

_MODELS_BY_NAME: dict[str, type[AttrsModel]] = {}
_MODEL_SCOPES: dict[type[AttrsModel], Scope] = {}
for scope, models in MODELS.items():
    for name, model_type in models.items():
        _MODELS_BY_NAME[name] = model_type
        _MODEL_SCOPES[model_type] = scope


def resolve_model(model: type[AttrsModel] | str) -> type[AttrsModel]:
    """Find a GeoSave attrs model by class or registered configuration name.

    Args:
        model: Model class or its name in the scope registry.

    Returns:
        The model class listed in `MODELS`.

    Raises:
        KeyError: No model is registered under this name.
        TypeError: The class is not listed in `MODELS`.

    Examples:
        >>> resolve_model("acdd")
        <class '...ACDD'>
    """
    if isinstance(model, str):
        try:
            return _MODELS_BY_NAME[model]
        except KeyError:
            raise KeyError(f"no attrs model is named {model!r}") from None
    if model in _MODEL_SCOPES:
        return model
    raise TypeError(f"{model!r} is not a GeoSave attrs model")


def model_scope(model: type[AttrsModel] | str) -> Scope:
    """Name the scope a model belongs to.

    Args:
        model: Model class or its name in the scope registry.

    Returns:
        `"dataset"`, `"variable"`, or `"coordinate"`.

    Raises:
        KeyError: No model is registered under this name.
        TypeError: The class is not listed in `MODELS`.
    """
    return _MODEL_SCOPES[resolve_model(model)]


@cache
def scope_keys(scope: Scope) -> frozenset[str]:
    """Return every attr key a model in `scope` writes."""
    keys: set[str] = set()
    for model in MODELS[scope].values():
        keys.update(model.attr_keys())
    return frozenset(keys)


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
    "Spectral",
    "StacItem",
    "StacMetadata",
    "StackedAttrs",
    "TimeSpec",
    "model_scope",
    "resolve_model",
    "scope_keys",
]
