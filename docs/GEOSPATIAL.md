# Geospatial Machine Learning Concepts & Preprocessing

Building machine learning models for Earth observation requires handling data that is intrinsically tied to physical space. Raw satellite imagery is rarely uniform, and directly feeding it into PyTorch without proper geospatial preprocessing will result in poor model performance and severe compute bottlenecks.

This document outlines the standard preprocessing pipeline, coordinate reference systems (CRS), and key geospatial operations.

---

## The Preprocessing Pipeline: Offline vs. Online

In geospatial ML, preprocessing is strictly divided into two phases to maximize GPU utilization and training speed.

### 1. Offline Preprocessing (Data Preparation)
This happens **once**, before training begins. Raw satellite data is heavy, compressed in slow formats, and spatially unaligned. Doing these operations on-the-fly during training starves the GPU of data.

**Goal:** Standardize geometry, resolution, and format.
**Tools:** `rasterio`, `gdal`, `xarray`, `rioxarray`, `stackstac`
**Outputs:** Cloud Optimized GeoTIFFs (COGs).

*Key Offline Operations:*
*   **Format Conversion:** Transcoding slow formats (`.jp2`, `.nc`) to COG (`.tif`) for rapid, random-access patch reading.
*   **CRS Alignment:** Reprojecting all datasets to a single, unified grid.
*   **Resampling:** Harmonizing pixel resolutions.
*   **Mosaicking & Clipping:** Stitching and cropping data to specific areas of interest.
*   **Cloud Masking:** Removing corrupted pixels via QA/cloud bitmasks.
*   **Feature Engineering:** Pre-calculating heavy indices (e.g., NDVI) as standalone bands.

### 2. Online Preprocessing (Data Augmentation & Normalization)
This happens **every epoch**, dynamically inside your PyTorch `LightningDataModule` or via `torchgeo.transforms`.

**Goal:** Transform spatial grids into mathematical tensors and prevent overfitting.
**Tools:** `torchgeo.transforms`, `torchvision`, `kornia`
**Outputs:** Float32 PyTorch Tensors `[B, C, H, W]`.

*Key Online Operations:*
*   **Tensor Conversion:** Stripping geospatial metadata and casting arrays to `torch.Tensor`.
*   **Normalization:** Scaling pixel values (e.g., Z-score normalization, Min-Max to `[0, 1]`). Calculated over the training split.
*   **Stochastic Augmentations:** Random crops, flips, rotations, and color jitter applied dynamically so the model sees varied data every epoch.
*   **NoData Handling:** Replacing `NaN` or `-9999` with fill values (e.g., `0`) before passing to the network. During loss calculation, these same pixels are masked out using `ignore_index`.

---

## Coordinate Reference Systems (CRS)

A CRS dictates how the 3D, curved surface of the Earth is flattened into a 2D computational grid. If your input imagery and ground-truth labels are in different CRSs, their pixels will not align, and physical distances will be distorted.

CRSs are typically referenced by their **EPSG** code.

### WGS84 (EPSG:4326) - Geographic
*   **Unit:** Degrees (Latitude, Longitude)
*   **Use Case:** Global mapping, GPS coordinates, web maps.
*   **Limitations:** Since it uses degrees, physical area changes depending on your distance from the equator. A $1^\circ \times 1^\circ$ pixel at the equator is much larger in square meters than one near the poles. It is generally **bad** for precise CNN/ViT training because a $10 \times 10$ pixel patch represents different physical sizes depending on global location.
*   **Example Coordinate:** `(48.8566, 2.3522)` (Paris)

### UTM (Universal Transverse Mercator) - Projected
*   **Unit:** Meters
*   **Use Case:** Local, highly accurate spatial modeling (e.g., Sentinel-2 delivery format).
*   **Mechanism:** Earth is divided into 60 narrow vertical zones. Within a specific zone, distortion is minimal, and a $10m \times 10m$ pixel is physically accurate. 
*   **Example CRSs:** `EPSG:32631` (UTM Zone 31N - Paris area)
*   **Example Coordinate:** `((452500.0, 5411500.0))` (Meters from the zone's internal origin)

---

## Key Geospatial Operations Explained

### CRS Alignment (Reprojection)
If your input image (Sentinel-2) is in `EPSG:32631` (Meters) and your label (Dynamic World) is in `EPSG:4326` (Degrees), TorchGeo must warp one to match the other. This requires complex trigonometry to interpolate pixel values onto the new grid. **Always align CRSs offline** to a common EPSG code to prevent massive processing overhead during training.

### Resampling
Adjusting the spatial resolution of an array to match a specific size.
*   *Example:* Sentinel-2 has bands at 10m, 20m, and 60m resolution. Before passing them as a unified `[C, H, W]` tensor, the 20m and 60m bands must be resampled (e.g., using nearest neighbor or bilinear interpolation) to match the 10m grid.

### Cloud Masking
Satellites often capture clouds, making the underlying pixels useless for surface segmentation. By using dedicated Scene Classification (SCL) or Quality Assurance (QA) bands provided by the satellite, you can identify cloud, shadow, and snow pixels. In the offline step, these pixels are permanently overwritten with a `NoData` value (like `-9999` or `NaN`) so the model doesn't try to learn from obstructed terrain.

### Mosaicking
Satellite imagery is collected and distributed in massive predefined tiles (e.g., 100km x 100km squares). If your area of interest (AOI) lies directly across the boundary of two adjacent tiles, you must download both and stitch them together into a single, seamless contiguous array.

### Clipping (Subsetting)
A single Sentinel-2 tile is over 1GB. If your training target is a small 5km x 5km city within that tile, you provide a bounding box (minimum X, minimum Y, maximum X, maximum Y) to "clip" the raster. This extracts only the necessary pixels, severely reducing disk footprint and memory requirements.

### Handling NoData during Inference
1.  **Input:** Deep learning architectures cannot multiply `NaN` values. Before inference, `NaN` pixels (like black image padding or cloud masks) are replaced with a safe fill value (`0`).
2.  **Prediction:** The network generates a prediction for the entire rectangular array, including the fill-value areas (which will produce garbage predictions).
3.  **Post-processing:** The original `NoData` mask is re-applied. The predicted values in those empty zones are overwritten back to `NoData` or an `ignore_index` class before computing validation metrics (e.g., Torchmetrics IoU).
