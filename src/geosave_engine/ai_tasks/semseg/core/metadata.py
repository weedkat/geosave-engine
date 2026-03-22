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
        # Validate and load metadata
        if isinstance(metadata, (str, Path)):
            metadata = yaml.load(open(metadata, 'r'), Loader=yaml.Loader)
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

        
    def rgb_to_class(self, mask_rgb):
        """ 
        Input is in (H, W, 3) format and returns (H, W) format.
        """
        if not isinstance(mask_rgb, np.ndarray):
            mask_rgb = np.array(mask_rgb)

        h, w, _ = mask_rgb.shape
        target = np.zeros((h, w), dtype=np.int64)

        for class_idx, item in self.class_dict.items():
            match = np.all(mask_rgb == item['rgb'], axis=-1)
            target[match] = class_idx

        return target
    
    def class_to_rgb(self, mask_class):
        """ 
        Input is in (H, W) format and output is in (H, W, 3) format
        """
        if not isinstance(mask_class, np.ndarray):
            mask_class = np.array(mask_class)

        h, w = mask_class.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, item in self.class_dict.items():
            rgb = np.array(item["rgb"], dtype=np.uint8)
            mask_rgb[mask_class == class_idx] = rgb

        return mask_rgb
    
    def get_class_dict(self, include_ignore_index=True):
        if not include_ignore_index:
            return {key: value for key, value in self.class_dict.items() if key != self.ignore_index}
        return self.class_dict
    
    def get_selected_bands(self):
        return self.get_bands(self.selected_bands)
    
    def get_bands(self, band_names):
        """Get list of band indices based on available_bands in metadata"""
        return [self.available_bands[name] for name in band_names]