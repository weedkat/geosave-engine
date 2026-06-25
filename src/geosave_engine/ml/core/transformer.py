import torch
import torch.nn as nn
import inspect

import kornia.augmentation as K

def build_augmentation_pipeline(augmentations: list, global_size: tuple[int, int] | int) -> list:
    """Recursively parses configuration dicts to build a list of Kornia objects."""
    aug_instances = []
    
    for aug in augmentations:
        aug_name = aug["name"]
        init_args = aug.get("init_args", {}).copy()
        
        if not hasattr(K, aug_name):
            raise ValueError(f"Unknown augmentation '{aug_name}' in Kornia.")
            
        aug_cls = getattr(K, aug_name)
        
        if aug_name == "AugmentationSequential":
            nested_augs = aug.get("augmentations", [])
            nested_pipeline_list = build_augmentation_pipeline(nested_augs, global_size)
            
            aug_instances.append(K.AugmentationSequential(*nested_pipeline_list, **init_args))
            continue

        sig = inspect.signature(aug_cls.__init__)
        if "size" in sig.parameters:
            init_args.setdefault("size", global_size)

        aug_instances.append(aug_cls(**init_args))
        
    return aug_instances

class ImageProcessor(nn.Module):
    """
    A class for processing images, including resizing, normalization, and augmentation.
    """

    def __init__(self, augmentations: list, size, mean, std):
        super().__init__()
        self.size = size
        self.mean = mean
        self.std = std

        augmentations_list = build_augmentation_pipeline(augmentations or [], size)

        has_normalize = any(isinstance(a, K.Normalize) for a in augmentations_list)
        if not has_normalize:
            augmentations_list.append(
                K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
            )

        self.augmentations = K.AugmentationSequential(
            *augmentations_list,
            data_keys=["input", "mask"],
        )
    
    def forward(self, img, mask=None):
        if self.training and mask is not None:
            img, _ = self.augmentations(img, mask)
        else:
            img = self.augmentations(img)
        return img