import torch
import torch.nn as nn
import kornia.augmentation as K
import inspect


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


class SemanticSegmentationWrapper(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        augmentations: list | None, 
        size: tuple[int, int] | int,
    ):
        super().__init__()
        self.model = model
        self.augmentations = augmentations or []

        pipeline_components = build_augmentation_pipeline(self.augmentations, size)
        self.random_augmentations = K.AugmentationSequential(
            *pipeline_components, 
            data_keys=["input", "mask"],
        )
        
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        """
        Processes standard forward flows.
        If training, expects both image (x) and mask (y) to be provided together.
        """
        if self.training and y is not None:
            x, y = self.random_augmentations(x, y)
            logits = self.model(x)
            return logits, y
                 
        return self.model(x)
