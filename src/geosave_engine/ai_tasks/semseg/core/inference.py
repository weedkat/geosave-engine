from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Tuple, Dict, List

from .transform import TransformsCompose
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
        
        # Transform configuration
        self.transform = TransformsCompose(self.transform_cfg, input_size=self.patch_size)
        self.model.model.eval()
    
    def preprocess(
        self,
        img_np: np.ndarray
    ):
        """
        Convert numpy array to normalized tensor batch.
        
        Args:
            image: Numpy array, either (H, W, C) or (B, H, W, C)

        Returns:
            img_tensor: (B, C, H, W) normalized tensor
            orig_shape: (H, W) tuple
        """
        # Ensure (B, H, W, C) format for single images
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)  # (H, W, 1)
        if img_np.ndim == 3:
            img_np = np.expand_dims(img_np, axis=0)  # (1, H, W, C)
        
        # Ensure uint8
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        
        # Process each image in batch through transforms
        img_tensors = []
        for i in range(img_np.shape[0]):
            img_tensor = self.transform(image=img_np[i])['image']
            img_tensors.append(img_tensor)
        
        img_tensor = torch.stack(img_tensors, dim=0)  # (B, C, H, W)
        
        return img_tensor
    
    def postprocess(
        self,
        logits: torch.Tensor
    ):
        """
        Convert model output tensor to numpy array of predicted labels.
        
        Args:
            logits: (B, C, H, W) logits output from model
            
        Returns:
            pred_np: (B, H, W) predicted class indices
        """
        pred = logits.argmax(dim=1).cpu().numpy()  # (B, C, H, W) -> (B, H, W)
        conf = logits.softmax(dim=1).cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        # Apply confidence threshold to reject uncertain predictions
        max_conf = np.max(conf, axis=-1)  # (B, H, W, C) -> (B, H, W)
        pred[max_conf < self.confidence_threshold] = self.reject_class

        return pred, conf
    
    def engine(self, images, mode, overlap_ratio):

        start_time = time.time()

        img_tensor = self.preprocess(images)

        with torch.no_grad():
            if mode == 'resize':
                output = self._infer_resize(img_tensor)
            elif mode == 'sliding_window':
                output = self._infer_sliding_window(img_tensor, overlap_ratio)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        
        pred, conf = self.postprocess(output)

        elapsed = time.time() - start_time
        
        self.model.log(f'[Inference] Mode: {mode}, Processed {img_tensor.shape} in {elapsed:.2f}s')
        
        return pred, conf
    
    def __call__(self, images):
        """
        Perform inference on input image(s).
        
        Args:
            images: Numpy array (H, W, C) or (B, H, W, C)
            mode: 'sliding_window' or 'resize'
            overlap_ratio: Overlap ratio for sliding window (default: from init)
            log: Whether to log processing information

        Returns:
            pred: (H, W) or (B, H, W) predicted labels
            conf: (H, W, C) or (B, H, W, C) confidence scores per class
            metadata: dict with processing info
        """
        assert 0.0 <= overlap_ratio < 1.0, "Overlap ratio must be in [0, 1)"
        assert mode in ['resize', 'sliding_window'], f"Invalid mode: {mode}"

        if isinstance(images, np.ndarray):
            return self.engine(images, mode, overlap_ratio)

        elif isinstance(images, DataLoader):
            preds, confs = [], []
            for batch in images:
                pred, conf = self.engine(batch, mode, overlap_ratio)
                preds.append(pred)
                confs.append(conf)

            pred = np.concatenate(preds, axis=0)
            conf = np.concatenate(confs, axis=0)
    
            return pred, conf
        
        else:
            raise ValueError("Input must be a numpy array or a DataLoader")
    
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