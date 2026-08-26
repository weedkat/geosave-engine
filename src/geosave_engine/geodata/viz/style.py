"""RenderStyle: how pixel values become colors. See RenderStyle for details."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RenderStyle(BaseModel):
    """Policy a view follows when it has no explicit instruction.

    Separate from `RenderHints`, which records what one raster means —
    this records how any raster is drawn. Pass one to reuse the same
    choices across a session; override a single view with hvplot keywords.

    Args:
        stretch: Percentiles the color range spans, low then high.
        sequential_cmap: Colormap for values on one side of zero.
        diverging_cmap: Colormap for values straddling zero.
        sample_cap: Most pixels read to derive a color range. An array
            larger than this is decimated, so an unbounded surface never
            has to be read whole to be drawn.
        aspect: Pixel aspect ratio. `1.0` keeps ground distances square;
            None lets the plot stretch to its frame.

    Raises:
        ValueError: `stretch` isn't an ascending pair inside 0–100.

    Examples:
        >>> style = RenderStyle(stretch=(5.0, 95.0), sequential_cmap="magma")
        >>> raster.explore(style=style)
    """

    model_config = ConfigDict(frozen=True)

    stretch: tuple[float, float] = (2.0, 98.0)
    sequential_cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"
    sample_cap: int = Field(default=512 * 512, gt=0)
    aspect: float | None = 1.0

    @model_validator(mode="after")
    def _validate_stretch(self) -> RenderStyle:
        """Check the stretch percentiles ascend and stay in range.

        Returns:
            This style, unchanged.

        Raises:
            ValueError: The pair isn't ascending, or falls outside 0–100.
        """
        low, high = self.stretch
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(f"stretch must be an ascending pair inside 0–100, got {self.stretch}")
        return self


DEFAULT_STYLE = RenderStyle()
