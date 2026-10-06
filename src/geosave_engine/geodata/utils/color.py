"""Palette colours shared by metadata, persistence, visualization, and training."""

from __future__ import annotations

from collections.abc import Sequence

type RGB = tuple[int, int, int]
type Palette = dict[int, RGB] | dict[int, str]


def parse_color(color: Sequence[int] | str) -> RGB:
    """Read one palette colour as an RGB triple.

    Args:
        color: ``"#RRGGBB"`` hex string (the ``#`` is optional), or three
            channels as a tuple or list, as JSON and YAML hand them back.

    Returns:
        ``(R, G, B)`` with each channel in ``[0, 255]``.

    Raises:
        ValueError: The colour is not six hex digits or three channels in range.

    Examples:
        >>> parse_color("#ff8000")
        (255, 128, 0)
        >>> parse_color([255, 128, 0])
        (255, 128, 0)
    """
    try:
        if isinstance(color, str):
            hexed = color.removeprefix("#")
            if len(hexed) != 6:
                raise ValueError
            rgb = tuple(int(hexed[i : i + 2], 16) for i in (0, 2, 4))
        else:
            rgb = tuple(int(channel) for channel in color)
        if len(rgb) != 3 or not all(0 <= channel <= 255 for channel in rgb):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"colour {color!r} is neither '#RRGGBB' nor three channels in [0, 255]"
        ) from None
    return rgb  # type: ignore[return-value]
