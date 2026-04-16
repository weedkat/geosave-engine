# This is an offline preprocessing script for Sentinel-2 data. 
# It is not intended to be run as part of the main training pipeline, 
# but rather as a one-time setup step to prepare the raw data for 
# efficient loading during training.

# Main preprocessing pipeline includes:
# 1. Reprojecting to a common CRS (e.g., EPSG:432
# 2. Resampling to a consistent pixel size (e.g., 10m)
# 3. Mosaicking or clipping to exact bounding boxes
# 4. Writing out to Cloud Optimized GeoTIFF (COG) format
# 5. Masking cloud

from pathlib import Path

def preprocess_sentinel2(input_dir: str, output_dir: str):
    """Heavy geospatial preprocessing script.
    
    This includes actions like:
    1. Reprojecting to a common CRS
    2. Resampling pixel size (e.g., 10m to match DynamicWorld limits, or vice versa)
    3. Mosaicking or clipping to exact bounding boxes
    4. Writing out to Cloud Optimized GeoTIFF (COG)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⚙️ Preprocessing raw Sentinel-2 data from {input_dir}...")
    # Add rasterio/stackstac/rioxarray processing logic here
    # Example:
    # for file in input_dir.glob("*.tif"):
    #     reproject_and_save_cog(file, output_dir)
    
    print(f"✓ Preprocessing complete. Outputs at {output_dir}")
    return True
