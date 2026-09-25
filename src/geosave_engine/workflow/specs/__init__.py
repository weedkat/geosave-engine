"""Portable declarations owned by ``model_spec.yaml``."""

from .base import Name, SpecModel, Text
from .call import CallSpec, CallValue, Ref
from .model import ModelSpec
from .postprocessing import PostprocessingSpec
from .preprocessing import OperationSpec
from .stage import StageSpec
from .sources import (
    AttrsRequirement,
    FieldRequirement,
    NamespaceRequirement,
    RasterRequirement,
)

__all__ = [
    "AttrsRequirement",
    "CallSpec",
    "CallValue",
    "FieldRequirement",
    "ModelSpec",
    "Name",
    "NamespaceRequirement",
    "OperationSpec",
    "PostprocessingSpec",
    "RasterRequirement",
    "Ref",
    "SpecModel",
    "StageSpec",
    "Text",
]
