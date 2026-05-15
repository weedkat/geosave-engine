import torch
import torch.nn.functional as F

# UPDATE: added padding because models are more accurate at the center
def infer_sliding_window(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    grid_size: int | tuple[int, int],
    device: torch.device | str,
    overlap_ratio: float = 0.5,
    pad_size: int = 64,
) -> torch.Tensor:
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")

    if isinstance(grid_size, tuple):
        grid_h, grid_w = grid_size
    else:
        grid_h = grid_w = grid_size

    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_size must be > 0")

    stride_h = int(grid_h * (1 - overlap_ratio))
    stride_w = int(grid_w * (1 - overlap_ratio))
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("overlap_ratio too high for the given grid_size")

    target_device = torch.device(device)
    b, c, h, w = img_tensor.shape

    # 1. Calculate padding (half the grid size ensures the image edges hit the center of a patch)
    pad_h, pad_w = pad_size, pad_size  # small padding, we have hann window

    # Pad input: (left, right, top, bottom)
    padded_img = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h), mode="reflect").to(target_device)
    _, _, pad_h_out, pad_w_out = padded_img.shape

    if grid_h > pad_h_out or grid_w > pad_w_out:
        raise ValueError(f"Grid size {grid_size} is too large for the padded image size {(pad_h_out, pad_w_out)}")

    # 2. Setup accumulators based on PADDED size
    final = torch.zeros(b, c, pad_h_out, pad_w_out, device=target_device, dtype=torch.float32)
    weight = torch.zeros(b, 1, pad_h_out, pad_w_out, device=target_device, dtype=torch.float32)

    # Setup Hann window
    window_weight_h = torch.hann_window(grid_h, periodic=False, device=target_device)
    window_weight_w = torch.hann_window(grid_w, periodic=False, device=target_device)
    window_2d = (window_weight_h.unsqueeze(1) * window_weight_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0) # shape (1, 1, grid_h, grid_w)

    # 3. Sliding window over the PADDED image
    row = 0
    while row < pad_h_out:
        col = 0
        while col < pad_w_out:
            # Handle edge snapping perfectly
            row_start = min(row, pad_h_out - grid_h)
            col_start = min(col, pad_w_out - grid_w)
            row_end = row_start + grid_h
            col_end = col_start + grid_w

            window = padded_img[:, :, row_start:row_end, col_start:col_end]
            pred = model(window)

            final[:, :, row_start:row_end, col_start:col_end] += pred * window_2d
            weight[:, :, row_start:row_end, col_start:col_end] += window_2d

            if col >= pad_w_out - grid_w:
                break
            col += stride_w

        if row >= pad_h_out - grid_h:
            break
        row += stride_h

    # 4. Normalize
    final /= weight.clamp(min=1e-6)

    # 5. Crop back to original size
    final = final[:, :, pad_h:pad_h+h, pad_w:pad_w+w]

    return final