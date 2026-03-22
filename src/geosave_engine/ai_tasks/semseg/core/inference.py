import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple, Dict, List

from lightning import LightningModule


class Inference:
    """
    Inference wrapper for semantic segmentation models.
    
    Supports two inference modes:
    1. 'resize': Direct inference with upscaling to original size
    2. 'sliding_window': Patch-based inference with overlap and stitching
    
    Args:
        model: Trained segmentation model
        device: 'auto' (GPU if available), 'cuda', or 'cpu'
        patch_size: Patch size of the backbone (default 14 for DINOv2)
        overlap_ratio: Overlap ratio for sliding window (0.0-1.0, default 0.5)
    """
    def __init__(
        self,
        module: LightningModule,
    ):
        self.module = module
        self.model = module.model


    def __call__(self, img_tensor, overlap_ratio=0.5, pred=True, conf=False, max_conf=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on input image(s).
        """
        start_time = time.time()

        logits = self._infer_sliding_window(img_tensor, overlap_ratio)

        elapsed = time.time() - start_time
        
        return logits
    
    def _infer_sliding_window(self, img_tensor: torch.Tensor, overlap_ratio: float = 0.5) -> torch.Tensor:
        """
        Sliding window inference matching the original evaluation method.
        
        Args:
            img_tensor: (B, C, H, W)
            
        Returns:
            output: (B, C, H, W) logits
        """
        model = self.model.model  # Access the underlying model if wrapped
        grid = self.patch_size
        b, _, h, w = img_tensor.shape
        final = torch.zeros(b, self.nclass, h, w).to(self.device)

        # Semantic segmentation tend to be more accurate on the center of patches
        # hann_window create a bell curve to weight the center more
        weight = torch.zeros(b, 1, h, w).to(self.device)
        window_weight = torch.hann_window(grid, periodic=False).to(self.device)
        window_2d = window_weight.unsqueeze(0) * window_weight.unsqueeze(1)
        
        row = 0
        while row < h:
            col = 0
            while col < w:
                # Extract window, handling edge cases
                row_end = min(row + grid, h)
                col_end = min(col + grid, w)
                
                window = img_tensor[:, :, row:row_end, col:col_end]
                
                # Run inference
                pred = model(window)
                
                # Accumulate logits
                final[:, :, row:row_end, col:col_end] += pred * window_2d
                weight[:, :, row:row_end, col:col_end] += window_2d
                
                # Move to next column
                if col >= w - grid:
                    break

                # Make sure to start at w - grid when near edge
                col = min(col + int(grid * overlap_ratio), w - grid)
            
            # Move to next row
            if row >= h - grid:
                break
            row = min(row + int(grid * overlap_ratio), h - grid)
        
        final /= weight

        return final