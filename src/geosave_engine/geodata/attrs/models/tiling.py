"""Where one tile sits in the raster it was cut from."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import field_validator

from geosave_engine.geodata.attrs.model import AttrsModel

if TYPE_CHECKING:
    import numpy as np
    from tiler import Merger, Tiler

# Padding that keeps every tile on one shape, so tiles match a model input spec.
type TilingMode = Literal["reflect", "edge", "constant", "wrap"]

# tiler.Merger's own window names, which weigh the pixels several tiles cover.
type StitchWindow = Literal[
    "boxcar",
    "triang",
    "blackman",
    "hamming",
    "hann",
    "bartlett",
    "parzen",
    "bohman",
    "blackmanharris",
    "nuttall",
    "barthann",
    "overlap-tile",
]


class Tiling(AttrsModel):
    """Locate one tile within the raster it was cut from.

    Every field is required: a partial stamp cannot place a tile or rebuild
    the raster around it.

    Args:
        source_id: Shared by every tile cut from one raster in one operation.
        tile_index: This tile's position in row-major order.
        source_shape: Height and width of the raster before cutting.
        tile_shape: Height and width of each tile.
        tile_overlap: Pixels neighbouring tiles share, as a non-negative count,
            a fraction in `[0, 1)`, or non-negative row-column counts.
        tile_padding: How the trailing edges were extended so every tile holds
            one shape.

    Raises:
        ValueError: A count is negative, a shape dimension is not positive, or
            a fractional overlap falls outside `[0, 1)`.

    Examples:
        >>> Tiling(
        ...     source_id="scene-001",
        ...     tile_index=3,
        ...     source_shape=(2048, 2048),
        ...     tile_shape=(512, 512),
        ...     tile_overlap=64,
        ...     tile_padding="reflect",
        ... )
    """

    NAME: ClassVar[str] = "tiling"

    source_id: str
    tile_index: int
    source_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    tile_overlap: int | float | tuple[int, int]
    tile_padding: TilingMode

    @property
    def count(self) -> int:
        """Return how many tiles the raster is cut into.

        Examples:
            >>> placement.count
            25
        """
        return len(self.tiler())

    @property
    def padded_shape(self) -> tuple[int, int]:
        """Return the height and width the raster is extended to before cutting.

        Equal to `source_shape` where the tiles divide it exactly.

        Returns:
            Shape the trailing tiles reach, never smaller than `source_shape`.

        Examples:
            >>> Tiling(..., source_shape=(100, 100), tile_shape=(64, 64)).padded_shape
            (128, 128)
        """
        layout = self.tiler()
        reach = layout.get_tile_bbox(len(layout) - 1)[1]
        return (int(reach[0]), int(reach[1]))

    @property
    def bounds(self) -> tuple[slice, slice]:
        """Return the row and column slices this tile covers.

        Measured against `padded_shape`, so a trailing tile's slice runs past
        `source_shape`.

        Returns:
            Row slice and column slice, usable on a Dataset or a GeoBox alike.

        Examples:
            >>> rows, columns = placement.bounds
            >>> padded.isel({"y": rows, "x": columns})
        """
        near, far = self.tiler().get_tile_bbox(self.tile_index)
        return (slice(int(near[0]), int(far[0])), slice(int(near[1]), int(far[1])))

    def tiles(self) -> Iterator[Self]:
        """Yield every tile placement in row-major order.

        Yields:
            This placement with each `tile_index` in turn.
        """
        for tile_index in range(len(self.tiler())):
            yield self.model_copy(update={"tile_index": tile_index})

    def tiler(self, channels: int = 0) -> Tiler:
        """Build the layout this placement describes.

        Args:
            channels: Length of one axis held out of the tiling, ahead of the
                spatial pair. Zero tiles the spatial axes alone.

        Returns:
            Layout over `source_shape`, cutting `tile_shape` tiles.

        Examples:
            >>> placement.tiler(channels=6).get_tile_bbox(3)
            (array([0, 48, 48]), array([6, 112, 112]))
        """
        from tiler import Tiler

        if not channels:
            return Tiler(
                data_shape=self.source_shape,
                tile_shape=self.tile_shape,
                overlap=self.tile_overlap,
                mode=self.tile_padding,
            )
        overlap = (
            (0, *self.tile_overlap)
            if isinstance(self.tile_overlap, tuple)
            else self.tile_overlap
        )
        return Tiler(
            data_shape=(channels, *self.source_shape),
            tile_shape=(channels, *self.tile_shape),
            overlap=overlap,
            mode=self.tile_padding,
            channel_dimension=0,
        )

    def merger(
        self, *, window: StitchWindow | np.ndarray | None = None, channels: int = 0
    ) -> Merger:
        """Build the accumulator that rebuilds this cut's raster.

        Args:
            window: Weighting applied across each tile before overlaps are
                averaged. None weighs every pixel alike, which rebuilds the
                source exactly.
            channels: Length of one axis held out of the tiling.

        Returns:
            Accumulator sized for the whole raster, every tile laid in by its
            own `tile_index`.

        Raises:
            ValueError: `window` names no window `tiler` knows, or a tapering
                window was asked for on a cut whose tiles do not overlap.

        Examples:
            >>> accumulator = placement.merger(window="hann", channels=6)
        """
        from tiler import Merger

        if isinstance(window, str) and window not in Merger.SUPPORTED_WINDOWS:
            raise ValueError(
                f"{window!r} is not a window tiler knows; pick one of "
                f"{sorted(Merger.SUPPORTED_WINDOWS)}"
            )
        if window is not None and not self.overlaps:
            raise ValueError(
                "a window weighs tiles against each other, which needs tiles cut "
                "with an overlap; cut with tile_overlap=... or rebuild without a window"
            )
        return Merger(self.tiler(channels=channels), window=window)

    @property
    def overlaps(self) -> bool:
        """Whether neighbouring tiles of this cut share any pixels."""
        if isinstance(self.tile_overlap, tuple):
            return any(self.tile_overlap)
        return self.tile_overlap > 0

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine tile placement, refusing disagreements.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side did not state it.

        Returns:
            Combined model, and the attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: A placement field disagrees or `models` is empty.
        """
        return cls._combine_fields(sides, must_agree=cls.model_fields)

    @field_validator("tile_index")
    @classmethod
    def _non_negative_tile_index(cls, value: int) -> int:
        """Reject a negative tile position."""
        if value < 0:
            raise ValueError("tile_index must be non-negative")
        return value

    @field_validator("source_shape", "tile_shape")
    @classmethod
    def _positive_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Reject a shape with a non-positive dimension."""
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("shape dimensions must be positive")
        return value

    @field_validator("tile_overlap")
    @classmethod
    def _overlap_in_range(
        cls, value: int | float | tuple[int, int]
    ) -> int | float | tuple[int, int]:
        """Reject a negative overlap count or an out-of-range fraction."""
        if isinstance(value, tuple):
            if value[0] < 0 or value[1] < 0:
                raise ValueError("overlap pixel counts must be non-negative")
        elif isinstance(value, int):
            if value < 0:
                raise ValueError("overlap pixel count must be non-negative")
        elif not 0 <= value < 1:
            raise ValueError("floating-point overlap must be in [0, 1)")
        return value
