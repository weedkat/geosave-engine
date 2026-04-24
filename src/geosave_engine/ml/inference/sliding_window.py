import torch
import torch.nn.functional as F  # noqa: N812 — universally accepted PyTorch alias

# UPDATE: added padding because models are more accurate at the center
def infer_sliding_window(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    grid_size: int,
    device: torch.device | str,
    overlap_ratio: float = 0.5,
    pad_size: int = 64,
) -> torch.Tensor:
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")

    grid = int(grid_size)
    if grid <= 0:
        raise ValueError("grid_size must be > 0")

    stride = int(grid * (1 - overlap_ratio))
    if stride <= 0:
        raise ValueError("overlap_ratio too high for the given grid_size")

    target_device = torch.device(device)
    b, c, h, w = img_tensor.shape

    # 1. Calculate padding (half the grid size ensures the image edges hit the center of a patch)
    pad_h, pad_w = pad_size, pad_size  # small padding, we have hann window

    # Pad input: (left, right, top, bottom)
    padded_img = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h), mode="reflect").to(target_device)
    _, _, pad_h_out, pad_w_out = padded_img.shape

    # 2. Setup accumulators based on PADDED size
    final: torch.Tensor | None = None
    weight = torch.zeros(b, 1, pad_h_out, pad_w_out, device=target_device)

    # Setup Hann window
    window_weight = torch.hann_window(grid, periodic=False, device=target_device)
    window_2d = (window_weight.unsqueeze(1) * window_weight.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    # 3. Sliding window over the PADDED image
    row = 0
    while row < pad_h_out:
        col = 0
        while col < pad_w_out:
            # Handle edge snapping perfectly
            row_start = min(row, pad_h_out - grid)
            col_start = min(col, pad_w_out - grid)
            row_end = row_start + grid
            col_end = col_start + grid

            window = padded_img[:, :, row_start:row_end, col_start:col_end]
            pred = model(window)

            if final is None:
                final = torch.zeros(
                    b,
                    pred.shape[1],
                    pad_h_out,
                    pad_w_out,
                    device=target_device,
                    dtype=pred.dtype,
                )

            final[:, :, row_start:row_end, col_start:col_end] += pred * window_2d
            weight[:, :, row_start:row_end, col_start:col_end] += window_2d

            if col >= pad_w_out - grid:
                break
            col += stride

        if row >= pad_h_out - grid:
            break
        row += stride

    if final is None:
        raise RuntimeError("sliding window inference produced no predictions")

    # 4. Normalize
    final /= weight.clamp(min=1e-6)

    # 5. Crop back to original size
    final = final[:, :, pad_h:pad_h+h, pad_w:pad_w+w]

    return final