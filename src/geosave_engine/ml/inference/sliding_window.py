from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_grid_size(grid_size: tuple[int, int] | int) -> tuple[int, int]:
    grid_h, grid_w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_size must be > 0")
    return grid_h, grid_w


def _patch_positions(
    padded_h: int,
    padded_w: int,
    grid_h: int,
    grid_w: int,
    overlap_ratio: float,
) -> list[tuple[int, int]]:
    """Top-left (row0, col0) corner of every patch covering a padded_h x padded_w area.

    Raises:
        ValueError: overlap_ratio out of [0, 1), or grid larger than the padded area.
    """
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")
    if grid_h > padded_h or grid_w > padded_w:
        raise ValueError(f"grid_size {(grid_h, grid_w)} too large for padded image {(padded_h, padded_w)}")

    stride_h = int(grid_h * (1 - overlap_ratio))
    stride_w = int(grid_w * (1 - overlap_ratio))
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("overlap_ratio too high for given grid_size")

    positions: list[tuple[int, int]] = []
    row = 0
    while True:
        col = 0
        while True:
            r0 = min(row, padded_h - grid_h)
            c0 = min(col, padded_w - grid_w)
            positions.append((r0, c0))
            if col >= padded_w - grid_w:
                break
            col += stride_w
        if row >= padded_h - grid_h:
            break
        row += stride_h
    return positions


def split_patches(
    image: torch.Tensor,
    grid_size: tuple[int, int] | int,
    overlap_ratio: float = 0.5,
    pad_size: int = 64,
) -> list[torch.Tensor]:
    """Reflect-pad and slice into overlapping patches for sliding-window inference.

    Pure tensor op — no model involved. Pair with `stitch_patches` (same
    `grid_size`/`overlap_ratio`/`pad_size`) to recombine whatever a caller
    does with these patches back into one full-resolution tensor.

    Args:
        image: `[B, C, H, W]` input tensor.
        grid_size: Patch size `(H, W)` or single int for square patches.
        overlap_ratio: Fraction of overlap between adjacent patches, `[0, 1)`.
        pad_size: Pixels of reflect-padding added on each side before patching.

    Returns:
        One contiguous `[B, C, grid_h, grid_w]` tensor per patch position,
        in row-major sweep order.

    Raises:
        ValueError: `overlap_ratio` out of `[0, 1)`, `grid_size <= 0`, or
            grid larger than the padded image.
    """
    grid_h, grid_w = _normalize_grid_size(grid_size)
    padded = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
    _, _, ph, pw = padded.shape
    positions = _patch_positions(ph, pw, grid_h, grid_w, overlap_ratio)
    return [padded[:, :, r0:r0 + grid_h, c0:c0 + grid_w].contiguous() for r0, c0 in positions]


def stitch_patches(
    predictions: list[torch.Tensor],
    original_shape: tuple[int, int],
    grid_size: tuple[int, int] | int,
    overlap_ratio: float = 0.5,
    pad_size: int = 64,
) -> torch.Tensor:
    """Hann-blend patch predictions back into one full-resolution tensor.

    Pure tensor op — recomputes the same patch positions `split_patches`
    used from the same args, no model involved.

    Args:
        predictions: One `[B, num_classes, grid_h, grid_w]` tensor per
            patch, in the same row-major sweep order `split_patches` returns.
        original_shape: `(H, W)` of the un-padded image `split_patches` was
            called on.
        grid_size: Patch size `(H, W)` or single int — same value passed to
            `split_patches`.
        overlap_ratio: Same value passed to `split_patches`.
        pad_size: Same value passed to `split_patches`.

    Returns:
        Blended `[B, num_classes, H, W]` tensor at original input resolution.

    Raises:
        ValueError: `overlap_ratio` out of `[0, 1)`, `grid_size <= 0`, grid
            larger than the padded image, or `predictions` is empty.
    """
    if not predictions:
        raise ValueError("stitch_patches: predictions must not be empty")

    grid_h, grid_w = _normalize_grid_size(grid_size)
    h, w = original_shape
    ph, pw = h + 2 * pad_size, w + 2 * pad_size
    positions = _patch_positions(ph, pw, grid_h, grid_w, overlap_ratio)

    b, n_classes = predictions[0].shape[:2]
    device = predictions[0].device
    final = torch.zeros(b, n_classes, ph, pw, device=device, dtype=torch.float32)
    weight = torch.zeros(b, 1, ph, pw, device=device, dtype=torch.float32)

    # [1, 1, grid_h, grid_w] — higher weight at patch center, fades to 0 at edges
    win_h = torch.hann_window(grid_h, periodic=False, device=device)
    win_w = torch.hann_window(grid_w, periodic=False, device=device)
    window_2d = (win_h.unsqueeze(1) * win_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    for (r0, c0), pred in zip(positions, predictions):
        r1, c1 = r0 + grid_h, c0 + grid_w
        final[:, :, r0:r1, c0:c1] += pred * window_2d
        weight[:, :, r0:r1, c0:c1] += window_2d

    final /= weight.clamp(min=1e-6)
    return final[:, :, pad_size:pad_size + h, pad_size:pad_size + w]
