"""Portable model requirements and Python/YAML call declarations."""

from .model import ModelSpec, OperationSpec, OutputSpec
from .references import Ref
from .requirements import (
    AttrsRequirement,
    FieldRequirement,
    NamespaceRequirement,
    RasterRequirement,
)

__all__ = [
    "AttrsRequirement",
    "FieldRequirement",
    "ModelSpec",
    "NamespaceRequirement",
    "OperationSpec",
    "OutputSpec",
    "RasterRequirement",
    "Ref",
]
