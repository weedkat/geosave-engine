"""Cuts a model reads its rasters through: frames along time, chips in space."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
import geopandas as gpd
from tiler import Tiler

from geosave_engine.geodata import cuts
from geosave_engine.geodata.cuts.frames import FrameMode

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
        >>> FramesSpec(length=4, stride=2, tolerance="10D").cut(windows)["id"].tolist()
        ['s0/frame-0', 's0/frame-1']
    """

    length: PositiveInt
    stride: PositiveInt | None = None
    tolerance: Text
    mode: FrameMode = "strict"

    def cut(self, windows: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Cut windows into frames, each group keeping the dates it observed on.

        Args:
            windows: Windows as `cuts.stacks` or `cuts.chips` lists them.

        Returns:
            One window per frame, in time order.

        Raises:
            ValueError: A window holds no sequence these frames fit.
        """
        return cuts.frames(
            windows,
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
        >>> ChipsSpec(size=224, overlap=32).cut(windows).iloc[0]["id"]
        's0/chip-0'
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

    def cut(self, windows: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Cut windows into chips of this size.

        Args:
            windows: Windows as `cuts.stacks` or `cuts.frames` lists them.

        Returns:
            One window per chip, each numbered within the window it was cut
            from.
        """
        return cuts.chips(windows, self.shape, overlap=self.overlap, mode=self.mode)

    def layout(self, shape: tuple[int, int]) -> tuple[Tiler, list[tuple[int, int]]]:
        """Lay these chips over a window of a given shape.

        Args:
            shape: Height and width of the window chips are cut from.

        Returns:
            The Tiler laying the chips and the halo around the window, the
            same every time, so a merge rebuilds what a cut used.
        """
        return cuts.layout(shape, self.shape, overlap=self.overlap, mode=self.mode)
