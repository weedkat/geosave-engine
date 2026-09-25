"""Model-owned requirements and YAML processing specifications."""

from .inference import (
    InferenceSpec,
    NormalizationSpec,
    SegmentationSpec,
    TensorInputSpec,
    TilingSpec,
    TimeWindowSpec,
)
from .model import ModelSpec
from .preprocessing import OperationSpec, PreprocessingSpec
from .requirements import (
    AttrsRequirement,
    FieldRequirement,
    NamespaceRequirement,
    RasterRequirement,
)

__all__ = [
    "AttrsRequirement",
    "FieldRequirement",
    "InferenceSpec",
    "ModelSpec",
    "NamespaceRequirement",
    "NormalizationSpec",
    "OperationSpec",
    "PreprocessingSpec",
    "RasterRequirement",
    "SegmentationSpec",
    "TensorInputSpec",
    "TilingSpec",
    "TimeWindowSpec",
]
