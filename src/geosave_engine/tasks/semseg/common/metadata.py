import numpy as np
import yaml
from pathlib import Path    

class MetadataInterpreter:
    """ 
    Handles metadata interpretation for class mapping and band selection.

    Args:
        metadata: Dict or path to YAML file containing 'class_dict', 'available_bands', and 'ignore_index' information.
    """
    def __init__(self, metadata):
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary or a path to a YAML file.")
        
        required_keys = ["available_bands", "selected_bands", "ignore_index", "class_dict", "input_size"]
        missing_keys = [key for key in required_keys if key not in metadata]
        
        if missing_keys:
            raise ValueError(f"Metadata is missing required keys: {missing_keys}")
    
        self.available_bands = metadata["available_bands"]
        self.selected_bands = metadata['selected_bands']
        self.ignore_index = metadata["ignore_index"]
        self.class_dict = metadata["class_dict"]
        self.input_size = metadata['input_size']
        self.metadata = metadata

        self.class_names = [item['name'] for key, item in self.class_dict.items() if key != self.ignore_index]
        self.nclass = len(self.class_names)
        self.in_channels = len(self.selected_bands)
        self.dataset = metadata.get('dataset', 'unknown')

        # --- OPTIMIZATION: Build Look-Up Tables (LUTs) in __init__ ---
        
        # 1. Setup Palette for class_to_rgb
        # Find the maximum class index to size our palette correctly
        max_idx = max(self.class_dict.keys())
        self.palette = np.zeros((max_idx + 1, 3), dtype=np.uint8)
        
        for cls_idx, item in self.class_dict.items():
            self.palette[cls_idx] = item['rgb']
            
        # 2. Setup Hash Table for rgb_to_class
        # 256^3 = 16,777,216 possible colors. 
        # We create a 1D array of this size (takes ~67MB of RAM) for O(1) lookups.
        self.color_to_class = np.full(256**3, self.ignore_index, dtype=np.int32)
        
        for cls_idx, item in self.class_dict.items():
            r, g, b = item['rgb']
            # Compress RGB into a single unique integer using bit shifting
            color_hash = (r << 16) | (g << 8) | b
            self.color_to_class[color_hash] = cls_idx

        
    def rgb_to_class(self, mask_rgb):
        """ 
        Vectorized RGB-to-Class conversion.
        Input: (H, W, 3) or batched (B, H, W, 3) format.
        Returns: (H, W) or (B, H, W) format.
        """
        # Ensure it's a numpy array and cast to uint32 to safely allow bitwise shifts
        mask_rgb = np.asarray(mask_rgb, dtype=np.uint32)

        # The ellipsis (...) allows this to work seamlessly with both (H,W,3) and (B,H,W,3)
        # Compress the RGB channels into a single integer hash
        # p.s i don't understand this voodoo trick
        color_hash = (mask_rgb[..., 0] << 16) | (mask_rgb[..., 1] << 8) | mask_rgb[..., 2]
        
        # Use the compressed integers as indices to grab the class IDs instantly
        return self.color_to_class[color_hash]
    
    
    def class_to_rgb(self, mask_class):
        """ 
        Vectorized Class-to-RGB conversion.
        Input: (H, W) or batched (B, H, W) format.
        Returns: (H, W, 3) or (B, H, W, 3) format.
        """
        # Ensure inputs are valid indices
        mask_class = np.asarray(mask_class, dtype=np.int32)

        # Advanced indexing instantly maps classes to colors and adds the RGB dimension.
        # Unknown/Ignore indices that exceed the palette size will raise an IndexError, 
        # which is standard/safe NumPy behavior.
        return self.palette[mask_class]
    
    
    def get_class_dict(self, include_ignore_index=True):
        if not include_ignore_index:
            return {key: value for key, value in self.class_dict.items() if key != self.ignore_index}
        return self.class_dict
    
    def get_selected_bands(self):
        return self.get_bands(self.selected_bands)
    
    def get_bands(self, band_names):
        """Get list of band indices based on available_bands in metadata"""
        return [self.available_bands[name] for name in band_names]