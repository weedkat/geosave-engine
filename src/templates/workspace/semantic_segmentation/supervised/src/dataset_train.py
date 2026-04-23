from __future__ import annotations

import numpy as np
import rasterio
import torch
import geopandas as gpd
from torch.utils.data import Dataset

from geosave_engine.ml.core.transform import TransformsCompose


class GeosaveDataset(Dataset):
    """Loads (input, label, mask) raster triples and remaps Pangaea labels to training IDs."""

    def __init__(
        self,
        paths: list[tuple[str, str, str]],
        class_meta: gpd.GeoDataFrame,
        transform: TransformsCompose | None = None,
    ):
        self.paths = paths
        self.transform = transform

        ignore_rows = class_meta[class_meta["ignore"].astype(bool)]
        self._ignore_index = int(ignore_rows["class_id"].iloc[0])

        max_src = int(class_meta["source_class_id"].max())
        lut = np.full(max_src + 1, self._ignore_index, dtype=np.int64)
        
        for _, row in class_meta.iterrows():
            src_id = int(row["source_class_id"])
            if src_id <= max_src:
                lut[src_id] = int(row["class_id"])

        self._label_lut = lut

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_path, label_path, mask_path = self.paths[index]

        with rasterio.open(input_path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)

        with rasterio.open(label_path) as src:
            label_raw = src.read(1)  # (H, W), Pangaea encoding

        with rasterio.open(mask_path) as src:
            cloud_mask = src.read(1).astype(bool)  # (H, W)

        label_clipped = np.clip(label_raw, 0, len(self._label_lut) - 1)
        label = self._label_lut[label_clipped]
        label[cloud_mask] = self._ignore_index
        label = label.astype(np.int64)

        image = np.moveaxis(image, 0, -1)  # (H, W, C) for albumentations

        if self.transform is not None:
            result = self.transform(image=image, mask=label)
            image = result["image"]
            label = result["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.moveaxis(image, -1, 0))
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)

        return {"image": image, "label": label}
