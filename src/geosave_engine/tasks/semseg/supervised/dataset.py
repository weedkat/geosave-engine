import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import rasterio
from PIL import Image
import numpy as np

from ..core.utils import extract_id

class BaseDataset(Dataset):
    def __init__(self, data_dir, data_df, metadata_cls, transform_fn, image_dir, label_dir, predict_mode=False):
        data_dir = Path(data_dir).resolve()
    
        self.df = data_df
        self.img_dir = data_dir / image_dir
        self.mask_dir = data_dir / label_dir
        self.mi = metadata_cls
        self.transform = transform_fn
        self.band_indices = self.mi.get_selected_bands()
        
        self.predict_mode = predict_mode

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        row = self.df.iloc[idx]
        
        img_path = self.img_dir / row['image']
        image, meta_profile = self._load_image(img_path)

        if self.predict_mode:
            result = self.transform(image=image)
            img_id = extract_id(img_path.stem)
            return result['image'], img_id, meta_profile
        
        if 'label' not in row or pd.isna(row['label']):
            raise ValueError(f"Label path is missing for index {idx} in training mode.")
        
        mask_path = self.mask_dir / row['label']
        mask = self._load_mask(mask_path)
        result = self.transform(image=image, mask=mask)

        # Ensure mask is int64 for loss function (CrossEntropyLoss expects Long)
        return result['image'], result['mask'].long()
        
    def _load_image(self, path):
        """Load image and extract spatial metadata if available."""
        path = Path(path)
        ext = path.suffix.lower()
        
        meta_profile = {}

        if ext in ['.tif', '.tiff']:
            # Use rasterio for TIFF files
            with rasterio.open(path) as src:
                meta_profile = src.profile.copy() 
                img = src.read([i + 1 for i in self.band_indices])  
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        else:
            img = Image.open(path)
            img = np.array(img)  
            
            if self.band_indices is not None and img.ndim >= 3:
                img = img[:, :, self.band_indices]
        
        # Return BOTH the image array and the metadata dictionary
        return img, meta_profile

    def _load_mask(self, path):
        """Load mask in either RGB format or index format"""
        path = Path(path)
        ext = path.suffix.lower()
        
        if ext in ['.tif', '.tiff']:
            # Use rasterio for TIFF files
            with rasterio.open(path) as src:
                m = src.read()  # (C, H, W)
                m = np.transpose(m, (1, 2, 0))  # (H, W, C) or (H, W, 1)
        else:
            # Use PIL for common formats
            m = Image.open(path)
            m = np.array(m)  # (H, W, C) or (H, W)
        
        # Detect if mask is RGB or index format
        if m.ndim == 3 and m.shape[2] == 3:
            # RGB format - convert to class indices
            mask = self.mi.rgb_to_class(m)
        elif m.ndim == 3 and m.shape[2] == 1:
            # Single channel with extra dimension - squeeze it
            mask = m.squeeze(-1).astype(np.int64)
        elif m.ndim == 2:
            # Already in index format
            mask = m.astype(np.int64)
        else:
            raise ValueError(f"Unexpected mask shape: {m.shape}")
        
        return mask
    
    def __len__(self):
        return len(self.df)
