import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import numpy as np

def load_yaml(config_path):
    """Load color configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def reverse_normalization(img, mean, std):
    """
    Reverse normalization for visualization
    
    x_norm = (x - mean) / std
    x = x_norm * std + mean
    """
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def is_normalized(
    img,
    atol=1e-6,
    mean_tol=1.0,
    std_tol=1.0,
):
    """
    Return True if image appears normalized using the selected mode.

    Notes:
    - Checks both dtype (must be floating) and statistics/range.
    - mode:
        * 'zero_one': min-max normalized to [0, 1]
        * 'standardized': z-score-like (mean~0, std~1)
        * 'either': accepts zero_one OR standardized
    """
    arr = np.asarray(img) # reject non float
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    if arr.size == 0: # empty array
        return False
    if not np.isfinite(arr).all(): # reject NaN or inf elements
        return False
    min_val = float(arr.min())
    max_val = float(arr.max())
    mean_val = float(arr.mean())
    std_val = float(arr.std())

    is_zero_one = min_val >= -atol and max_val <= 1.0 + atol
    is_standardized = abs(mean_val) <= mean_tol and abs(std_val - 1.0) <= std_tol

    return is_zero_one or is_standardized

class DrawSegmentation:
    def __init__(self, metadata):
        self.class_dict = load_yaml(metadata)['class_dict']
    
    def set_reverse_norm(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask, alpha=0.3, figsize=(5, 4)):
        """
        Display image with mask overlay and legend using actual RGB colors from class_dict.
        """
        if isinstance(img, torch.Tensor):
            if is_normalized(img) and hasattr(self, 'mean') and hasattr(self, 'std'):
                img = reverse_normalization(img, self.mean, self.std)
            img = img.numpy().transpose(1, 2, 0)
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)

        # Create colored mask using class_dict colors
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        unique_labels = np.unique(mask)
        for class_id in unique_labels:
            if class_id in self.class_dict:
                rgb = self.class_dict[class_id]['rgb']
                # Normalize RGB values to [0, 1]
                normalized_rgb = tuple(c / 255.0 for c in rgb)
                colored_mask[mask == class_id] = normalized_rgb
        ax.imshow(colored_mask, alpha=alpha)

        handles = []
        for class_id in unique_labels:
            if class_id in self.class_dict:
                label_text = self.class_dict[class_id]['name']
                rgb = self.class_dict[class_id]['rgb']
                normalized_rgb = tuple(c / 255.0 for c in rgb)
                patch = mpatches.Patch(color=normalized_rgb, label=label_text)
                handles.append(patch)

        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.axis('off')
        plt.tight_layout()
        plt.show()