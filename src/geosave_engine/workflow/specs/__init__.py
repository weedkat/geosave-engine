"""Portable declarations owned by ``model_spec.yaml``."""

from .base import Name, SpecModel, Text
from .model import ModelSpec
from .postprocessing import PostprocessingSpec
from .preprocessing import OperationSpec, Ref
from .sources import (
    AttrsRequirement,
    FieldRequirement,
    NamespaceRequirement,
    RasterRequirement,
)

__all__ = [
    "AttrsRequirement",
    "FieldRequirement",
    "ModelSpec",
    "Name",
    "NamespaceRequirement",
    "OperationSpec",
    "PostprocessingSpec",
    "RasterRequirement",
    "Ref",
    "SpecModel",
    "Text",
]
