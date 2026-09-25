"""Model argument encoding, spatial tiling, and output interpretation settings."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from geosave_engine.geodata.attrs import Legend
from geosave_engine.geodata.transform.tiling import StitchWindow, TilingMode

import pandas as pd

from .base import Name, RasterName, SpecModel, Text, unique


class NormalizationSpec(SpecModel):
    """Per-channel means and positive standard deviations in channel order."""

    mean: Annotated[tuple[float, ...], Field(min_length=1)]
    std: Annotated[tuple[Annotated[float, Field(gt=0)], ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _validate_channel_counts(self) -> Self:
        if len(self.mean) != len(self.std):
            raise ValueError("Normalization mean and std must have the same length")
        return self


class TensorInputSpec(SpecModel):
    """Encode a prepared raster as one named model argument.

    Args:
        raster: RasterName produced by preprocessing.
        variables: Explicit channel selection; None uses the prepared order.
        layout: Unbatched tensor axis order, using native spatial axis roles.
        dtype: Floating-point tensor dtype.
        normalize: Optional channel normalization after conversion.
    """

    raster: RasterName
    variables: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    layout: Literal["CHW", "TCHW"] = "CHW"
    dtype: Literal["float16", "bfloat16", "float32", "float64"] = "float32"
    normalize: NormalizationSpec | None = None

    @model_validator(mode="after")
    def _validate_variables(self) -> Self:
        if self.variables is not None:
            unique(self.variables, "input variables")
        return self


class TilingSpec(SpecModel):
    """Spatial tile geometry and merger settings on a prepared reference raster.

    Args:
        raster: Prepared raster providing the common grid.
        tile_shape: Height and width in pixels.
        overlap: Pixels shared between neighboring tiles.
        mode: Padding mode supported by Tiles.
        window: Merger weighting window; None weighs tiles uniformly.
    """

    raster: RasterName
    tile_shape: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    overlap: Annotated[int, Field(ge=0)] = 0
    mode: TilingMode = "reflect"
    window: StitchWindow | None = None

    @model_validator(mode="after")
    def _validate_overlap(self) -> Self:
        if self.overlap >= min(self.tile_shape):
            raise ValueError("Tile overlap must be smaller than both tile dimensions")
        if self.window is not None and not self.overlap:
            raise ValueError("A merge window requires positive tile overlap")
        return self


class SegmentationSpec(SpecModel):
    """Interpret merged multiclass logits with optional per-class thresholds.

    Args:
        method: Select the existing segmentation interpretation.
        classes: Native Legend tokens in output-channel order; positions are
            class IDs. Use names without whitespace, such as dry_land.
        thresholds: Optional confidence threshold for each class.
        ignore_index: Label for masked or low-confidence pixels.
    """

    method: Literal["segmentation"] = "segmentation"
    classes: Annotated[tuple[Text, ...], Field(min_length=2)]
    thresholds: tuple[Annotated[float, Field(ge=0, le=1)], ...] | None = None
    ignore_index: Annotated[int, Field(ge=0)] = 255

    @model_validator(mode="after")
    def _validate_classes(self) -> Self:
        unique(self.classes, "classes")
        Legend(class_map=dict(enumerate(self.classes)))
        if self.thresholds is not None and len(self.thresholds) != len(self.classes):
            raise ValueError("Thresholds must have one entry per class")
        if self.ignore_index < len(self.classes):
            raise ValueError("ignore_index must not collide with a class ID")
        return self


class TimeWindowSpec(SpecModel):
    """Sample temporal stacks using native window_stack semantics."""

    size: Annotated[int, Field(gt=0)]
    tolerance: str
    stride: Annotated[int, Field(gt=0)] | None = None
    mode: Literal["strict", "drop"] = "strict"

    @model_validator(mode="after")
    def _validate_tolerance(self) -> Self:
        try:
            tolerance = pd.Timedelta(self.tolerance)
        except (TypeError, ValueError) as error:
            raise ValueError("tolerance must be a valid duration") from error
        if pd.isna(tolerance) or tolerance <= pd.Timedelta(0):
            raise ValueError("tolerance must be a positive duration")
        return self


class InferenceSpec(SpecModel):
    """Describe model inputs, tiling, and interpretation independently of a run.

    Args:
        inputs: Forward argument names mapped to their raster encodings.
        tiling: Shared spatial tiling and merger configuration.
        time_window: Optional temporal sampling before spatial tiling.
    """

    inputs: Annotated[dict[Name, TensorInputSpec], Field(min_length=1)]
    tiling: TilingSpec
    time_window: TimeWindowSpec | None = None
