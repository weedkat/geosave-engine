from __future__ import annotations

import numpy as np
import torch

Palette = dict[int, tuple[int, int, int]] | dict[int, str]


def _parse_color(color: tuple[int, int, int] | str) -> tuple[int, int, int]:
    if isinstance(color, str):
        h = color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return color


def colorize(
    mask: np.ndarray | torch.Tensor,
    palette: Palette,
    default: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Map a class-index mask to an RGB image.

    Args:
        mask: ``(H, W)`` array of integer class indices. Accepts numpy array or
            torch Tensor; tensors are converted to numpy automatically.
        palette: Mapping from class index to RGB tuple ``(R, G, B)`` or hex
            string ``"#RRGGBB"``. Values are clamped to ``[0, 255]``.
        default: RGB colour for indices absent from ``palette``. Defaults to
            black ``(0, 0, 0)``.

    Returns:
        ``(H, W, 3)`` uint8 numpy array suitable for JPEG/PNG saving or logger
        consumption.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = np.asarray(mask)
    h, w = mask.shape
    rgb = np.full((h, w, 3), default, dtype=np.uint8)

    parsed: dict[int, tuple[int, int, int]] = {
        idx: _parse_color(color) for idx, color in palette.items()
    }

    for idx, color in parsed.items():
        rgb[mask == idx] = color

    return rgb
