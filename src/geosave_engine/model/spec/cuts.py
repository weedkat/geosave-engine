"""Cuts a model reads its rasters through: frames along time, chips in space."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
import xarray as xr
from tiler import Tiler

from geosave_engine.geodata.transform.time import FrameMode, stack_frames

from .base import SpecModel, Text


type TilingMode = Literal["reflect", "edge", "constant", "wrap"]
type StitchWindow = Literal[
    "boxcar",
    "hamming",
    "triang",
    "bartlett",
    "hann",
    "barthann",
    "bohman",
    "parzen",
    "blackman",
    "nuttall",
    "blackmanharris",
    "overlap-tile",
]


class FramesSpec(SpecModel):
    """Declare how a sample's time axis is cut into frames.

    Args:
        length: Instants each frame holds.
        stride: Instants between the starts of consecutive frames. None
            strides by `length`.
        tolerance: How far an instant may reach for a scene, such as `"10D"`.
        mode: What a frame covering an instant some layer missed does.

    Examples:
        >>> FramesSpec(length=4, stride=2, tolerance="10D").cut(sample)
        (<xarray.DataTree> ..., <xarray.DataTree> ...)
    """

    length: PositiveInt
    stride: PositiveInt | None = None
    tolerance: Text
    mode: FrameMode = "strict"

    def cut(self, sample: xr.DataTree) -> tuple[xr.DataTree, ...]:
        """Cut a stack into frames, each layer keeping the dates it observed on.

        Args:
            sample: Stack whose time-spanning layers carry a `time` coordinate.

        Returns:
            One stack per frame, in time order.

        Raises:
            ValueError: The stack holds no sequence these frames fit.
        """
        return stack_frames(
            sample,
            self.length,
            tolerance=self.tolerance,
            stride=self.stride,
            mode=self.mode,
        )


class ChipsSpec(SpecModel):
    """Declare how a raster is cut into chips and how results are merged back.

    Args:
        size: Chip side in pixels, or height and width.
        overlap: Pixels neighbouring chips share.
        mode: How a chip reaching past the raster is filled.
        window: How overlapping chips are weighed when merged. None weighs
            every chip alike.

    Raises:
        ValueError: Overlap cannot provide coverage for the selected window.

    Examples:
        >>> tiler = ChipsSpec(size=224, overlap=32).tiler((600, 800))
        >>> tuple(tiler.tile_shape)
        (224, 224)
    """

    size: PositiveInt | Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]
    overlap: NonNegativeInt = 0
    mode: TilingMode = "reflect"
    window: StitchWindow | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if self.window is not None and not self.overlap:
            raise ValueError(
                f"window {self.window!r} weighs overlapping chips; set overlap"
            )
        # Zero or negative edge weights need two pixels of shared support.
        if self.overlap == 1 and self.window in (
            "hann",
            "bartlett",
            "barthann",
            "bohman",
            "blackman",
            "overlap-tile",
        ):
            raise ValueError(
                f"window {self.window!r} needs at least 2 pixels of overlap to cover seams"
            )
        return self

    @property
    def shape(self) -> tuple[int, int]:
        """Return the chip height and width."""
        if isinstance(self.size, int):
            return self.size, self.size
        height, width = self.size
        return height, width

    def tiler(self, shape: tuple[int, int]) -> Tiler:
        """Build a native spatial Tiler from pixel dimensions.

        Args:
            shape: Parent height and width, before any optional halo padding.

        Returns:
            Native Tiler using this recipe. Constant padding represents missing
            pixels with NaN. Halo setup and reference construction are separate.
        """
        return Tiler(
            shape,
            self.shape,
            overlap=self.overlap,
            mode=self.mode,
            constant_value=float("nan"),
        )
