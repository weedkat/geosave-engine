from __future__ import annotations

from typing import Callable, Any

import torch
import torch.nn.functional as F


def sliding_window_inference(
    model_fn: Callable[[torch.Tensor], Any],
    img: torch.Tensor,
    grid_size: tuple[int, int] | int,
    overlap_ratio: float = 0.5,
    pad_size: int = 64,
) -> torch.Tensor:
    """Run sliding-window inference with Hann-blended patch accumulation.

    Pads the image, sweeps patches through ``model_fn``, and accumulates
    predictions in-place with Hann weighting. Memory cost is ``O(1)`` patches
    (no intermediate list).

    Args:
        model_fn: Callable that takes a patch ``[B, C, grid_h, grid_w]`` and
            returns logits ``[B, num_classes, grid_h, grid_w]``.
        img: Input tensor ``[B, C, H, W]``.
        grid_size: Patch size ``(H, W)`` or single int for square patches.
        overlap_ratio: Fraction of overlap between adjacent patches. Must be in ``[0, 1)``.
        pad_size: Pixels of reflect-padding added on each side before patching.

    Returns:
        Blended logits ``[B, num_classes, H, W]`` at original input resolution.

    Raises:
        ValueError: If ``overlap_ratio`` out of ``[0, 1)``, ``grid_size <= 0``,
            or grid larger than padded image.
    """
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")

    grid_h, grid_w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_size must be > 0")

    stride_h = int(grid_h * (1 - overlap_ratio))
    stride_w = int(grid_w * (1 - overlap_ratio))
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("overlap_ratio too high for given grid_size")

    b, _, h, w = img.shape
    padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
    _, _, ph, pw = padded.shape

    if grid_h > ph or grid_w > pw:
        raise ValueError(f"grid_size {(grid_h, grid_w)} too large for padded image {(ph, pw)}")

    # Run first patch to infer num_classes and device
    first_pred = model_fn(padded[:, :, :grid_h, :grid_w])
    n_classes = first_pred.shape[1]
    device = first_pred.device

    final = torch.zeros(b, n_classes, ph, pw, device=device, dtype=torch.float32)
    weight = torch.zeros(b, 1, ph, pw, device=device, dtype=torch.float32)

    # [1, 1, grid_h, grid_w] — higher weight at patch center, fades to 0 at edges
    win_h = torch.hann_window(grid_h, periodic=False, device=device)
    win_w = torch.hann_window(grid_w, periodic=False, device=device)
    window_2d = (win_h.unsqueeze(1) * win_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    row = 0
    while row < ph:
        col = 0
        while col < pw:
            r0 = min(row, ph - grid_h)
            c0 = min(col, pw - grid_w)
            r1, c1 = r0 + grid_h, c0 + grid_w

            # reuse first patch prediction instead of re-running model
            if r0 == 0 and c0 == 0:
                pred = first_pred
            else:
                pred = model_fn(padded[:, :, r0:r1, c0:c1])

            final[:, :, r0:r1, c0:c1] += pred * window_2d
            weight[:, :, r0:r1, c0:c1] += window_2d

            if col >= pw - grid_w:
                break
            col += stride_w
        if row >= ph - grid_h:
            break
        row += stride_h

    final /= weight.clamp(min=1e-6)
    return final[:, :, pad_size:pad_size + h, pad_size:pad_size + w]
