"""Palette values shared by metadata, persistence, and visualization."""

Palette = dict[int, tuple[int, int, int]] | dict[int, str]


def parse_color(color: tuple[int, int, int] | str) -> tuple[int, int, int]:
    """Read one palette colour as an RGB triple.

    Args:
        color: ``"#RRGGBB"`` hex string, or an ``(R, G, B)`` triple returned
            unchanged.

    Returns:
        ``(R, G, B)`` with each channel in ``[0, 255]``.
    """
    if isinstance(color, str):
        h = color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return color
