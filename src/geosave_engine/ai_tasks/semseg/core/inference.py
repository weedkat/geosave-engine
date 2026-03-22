import torch
import torch.nn.functional as F
from lightning import LightningModule

def infer_sliding_window(module: LightningModule, img_tensor: torch.Tensor, overlap_ratio: float = 0.5) -> torch.Tensor:
    model = module.model
    grid = module.input_size
    b, c, h, w = img_tensor.shape
    
    # 1. Calculate padding (half the grid size ensures the image edges hit the center of a patch)
    pad_h, pad_w = grid // 2, grid // 2
    
    # Pad input: (left, right, top, bottom)
    padded_img = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    _, _, pad_h_out, pad_w_out = padded_img.shape

    # 2. Setup accumulators based on PADDED size
    final = torch.zeros(b, module.nclass, pad_h_out, pad_w_out, device=module.device)
    weight = torch.zeros(b, 1, pad_h_out, pad_w_out, device=module.device)

    # Setup Hann window
    window_weight = torch.hann_window(grid, periodic=False).to(module.device)
    window_2d = (window_weight.unsqueeze(1) * window_weight.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    stride = int(grid * (1 - overlap_ratio))
    
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
            
            final[:, :, row_start:row_end, col_start:col_end] += pred * window_2d
            weight[:, :, row_start:row_end, col_start:col_end] += window_2d
            
            if col >= pad_w_out - grid:
                break
            col += stride
            
        if row >= pad_h_out - grid:
            break
        row += stride
    
    # 4. Normalize
    final /= weight.clamp(min=1e-6)

    # 5. Crop back to original size
    final = final[:, :, pad_h:pad_h+h, pad_w:pad_w+w]

    return final