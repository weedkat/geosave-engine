from __future__ import annotations

import inspect

import torch
import kornia.augmentation as K

from typing import Literal


def build_augmentation_pipeline(augmentations: list[dict], size: tuple[int, int] | int) -> list:
    """Build list of Kornia augmentation instances from config dicts.

    Recursively handles ``AugmentationSequential``. Injects ``size`` into any
    augmentation whose ``__init__`` accepts it, if not already set.

    Args:
        augmentations: List of dicts with ``"name"`` and optional ``"init_args"``.
        size: Default spatial size injected into size-aware augmentations.

    Returns:
        List of instantiated Kornia augmentation objects.

    Raises:
        ValueError: If ``name`` is not a valid Kornia augmentation class.
    """
    aug_instances = []
    for aug in augmentations:
        aug_name = aug["name"]
        init_args = aug.get("init_args", {}).copy()
        if not hasattr(K, aug_name):
            raise ValueError(f"Unknown augmentation '{aug_name}' in Kornia.")
        aug_cls = getattr(K, aug_name)
        if aug_name == "AugmentationSequential":
            nested = build_augmentation_pipeline(aug.get("augmentations", []), size)
            aug_instances.append(K.AugmentationSequential(*nested, **init_args))
            continue
        sig = inspect.signature(aug_cls.__init__)
        if "size" in sig.parameters:
            init_args.setdefault("size", size)
        aug_instances.append(aug_cls(**init_args))
    return aug_instances


DataKey = Literal[
    "input",       # Standard image
    "image",       # Alias for input
    "mask",        # Spatial mask (segmentation)
    "bbox",        # Defaults to xyxy (Pascal VOC format)
    "bbox_xyxy",   # Pascal VOC format
    "bbox_xywh",   # COCO format
    "bbox_yolo",   # CUSTOM: YOLO format (intercepted and processed internally)
    "keypoints",   # 2D point coordinates
    "class",       # 1D class integer
    "label"        # Alias for class
]

# (Keep your build_augmentation_pipeline function here)

def yolo_to_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert normalized YOLO [cx, cy, w, h] to absolute [x_min, y_min, x_max, y_max]."""
    cx = boxes[..., 0] * width
    cy = boxes[..., 1] * height
    bw = boxes[..., 2] * width
    bh = boxes[..., 3] * height
    
    x_min = cx - (bw / 2)
    y_min = cy - (bh / 2)
    x_max = cx + (bw / 2)
    y_max = cy + (bh / 2)
    
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def xyxy_to_yolo(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert absolute [x_min, y_min, x_max, y_max] to normalized YOLO [cx, cy, w, h]."""
    bw = boxes[..., 2] - boxes[..., 0]
    bh = boxes[..., 3] - boxes[..., 1]
    cx = boxes[..., 0] + (bw / 2)
    cy = boxes[..., 1] + (bh / 2)
    
    return torch.stack([cx / width, cy / height, bw / width, bh / height], dim=-1)


class ImageAugmenter(torch.nn.Module):
    """Universal stochastic Kornia transforms for training.
    
    Dynamically routes inputs based on `data_keys`. Supports classification, 
    semantic segmentation, pixel-wise regression, and object detection.
    """

    def __init__(
        self,
        augmentations: list[dict],
        size: tuple[int, int] | int,
        data_keys: list[DataKey] | None = None,
    ) -> None:
        super().__init__()
        
        data_keys = data_keys or ["input"]
        self.yolo_indices = [i for i, k in enumerate(data_keys) if k == "bbox_yolo"]
        
        if self.yolo_indices:
            try:
                self.img_idx = next(i for i, k in enumerate(data_keys) if k in ("input", "image"))
            except StopIteration:
                raise ValueError("Using 'bbox_yolo' requires an 'input' or 'image' tensor in data_keys.")

        # Map custom YOLO keys to Kornia's native xyxy keys
        kornia_keys = ["bbox_xyxy" if k == "bbox_yolo" else k for k in data_keys]
        aug_list = build_augmentation_pipeline(augmentations, size)
        
        self.pipeline: K.AugmentationSequential | None = (
            K.AugmentationSequential(*aug_list, data_keys=kornia_keys)
            if aug_list else None
        )

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Apply stochastic pipeline to input tensors."""
        if self.pipeline is None:
            if not args:
                return ()
            return args[0] if len(args) == 1 else args

        # Fast path: No YOLO bounding boxes to process
        if not self.yolo_indices:
            return self.pipeline(*args)

        # 1. Pre-process YOLO -> XYXY
        args_list = list(args)
        _, _, h, w = args_list[self.img_idx].shape
        
        for i in self.yolo_indices:
            args_list[i] = yolo_to_xyxy(args_list[i], height=h, width=w)

        # 2. Run Kornia Pipeline
        out = self.pipeline(*args_list)

        # 3. Post-process XYXY -> YOLO
        # Normalize Kornia's output to a mutable list based on input length
        out_list = [out] if len(args) == 1 else list(out)
        _, _, new_h, new_w = out_list[self.img_idx].shape
        
        for i in self.yolo_indices:
            out_list[i] = xyxy_to_yolo(out_list[i], height=new_h, width=new_w)

        return out_list[0] if len(args) == 1 else tuple(out_list)