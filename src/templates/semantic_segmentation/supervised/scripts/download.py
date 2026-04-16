from pathlib import Path

from download_dw import download_dynamicworld
from extract_dw import extract_and_rearrange_dw
from download_sentinel2 import download_sentinel2_from_xlsx
from preprocess_sentinel2 import preprocess_sentinel2


def main():
    base_data_dir = Path("data")
    base_data_dir.mkdir(exist_ok=True)
    
    # 1. DynamicWorld
    dw_file_id = "YOUR_DW_FILE_ID"
    dw_zip = base_data_dir / "dynamicworld.zip"
    dw_extract = base_data_dir / "dynamicworld_extracted"
    
    print("--- 1. Processing DynamicWorld ---")
    download_dynamicworld(file_id=dw_file_id, output_path=dw_zip)
    extract_and_rearrange_dw(zip_path=dw_zip, extract_to=dw_extract)
    
    # 2. Sentinel-2
    s2_xlsx = base_data_dir / "target_tiles.xlsx"
    s2_raw = base_data_dir / "sentinel2_raw"
    s2_processed = base_data_dir / "sentinel2_processed"
    
    print("\n--- 2. Processing Sentinel-2 ---")
    download_sentinel2_from_xlsx(xlsx_path=s2_xlsx, output_dir=s2_raw)
    preprocess_sentinel2(input_dir=s2_raw, output_dir=s2_processed)
    
    print("\n🎉 All dataset preparations configured.")


if __name__ == "__main__":
    main()
