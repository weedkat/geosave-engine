import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import rasterio
from PIL import Image
import numpy as np

from ..core.utils import extract_id

class SemSegDataset(Dataset):
    """
    Generic dataset handler for various image formats (TIF, JPG, PNG, etc.)
    Supports masks in both RGB format and index format.

    Args:
        data_csv: Path to CSV file (relative to cwd or absolute)
        root_dir: Root directory of dataset (relative to cwd or absolute)
        metadata: Path to YAML metadata (relative to cwd or absolute)
        bands: List of band names to be used e.g. ['R', 'G', 'B'] or ['B1', 'B2', 'B3']
    """
    def __init__(self, data_dir, data_df, metadata_interpreter, transform, image_dir, label_dir):
        """
        Args:
            data_csv (str): Path to the CSV file containing image and mask paths.
            data_dir (str): Directory containing the dataset.
            metadata_interpreter (MetadataInterpreter): Instance of MetadataInterpreter.
            image_dir (str): Directory containing the images.
            label_dir (str): Directory containing the labels.
        """
        # Resolve relative paths from current working directory
        data_dir = Path(data_dir).resolve()
    
        self.df = data_df
        self.img_dir = data_dir / image_dir
        self.mask_dir = data_dir / label_dir
        self.mi = metadata_interpreter
        self.transform = transform
        self.band_indices = self.mi.get_selected_bands()

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        row = self.df.iloc[idx]
        
        img_path = self.img_dir / row['image']
        image = self._load_image(img_path)
        
        if 'label' not in row or pd.isna(row['label']): # predict mode, no labels available
            result = self.transform(image=image)
            img_id = extract_id(img_path.stem)
            return result['image'], img_id
        
        mask_path = self.mask_dir / row['label']
        mask = self._load_mask(mask_path)
        result = self.transform(image=image, mask=mask)

        # Ensure mask is int64 for loss function (CrossEntropyLoss expects Long)
        return result['image'], result['mask'].long()
        
    def _load_image(self, path):
        """Load image from various formats (TIF, JPG, PNG, etc.) and select bands if specified"""
        path = Path(path)
        ext = path.suffix.lower()

        if ext in ['.tif', '.tiff']:
            # Use rasterio for TIFF files
            with rasterio.open(path) as src:
                img = src.read([i + 1 for i in self.band_indices])  # (C, H, W), rasterio bands are 1-indexed
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        else:
            # Use PIL for common formats (JPG, PNG, etc.)
            img = Image.open(path)
            img = np.array(img)  # Already in (H, W, C) or (H, W)
            # Select bands if specified
            if self.band_indices is not None and img.ndim >= 3:
                img = img[:, :, self.band_indices]
        
        return img

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
