import zipfile
import os
from pathlib import Path

def extract_and_rearrange_dw(zip_path: str, extract_to: str, skip_if_extracted: bool = True) -> bool:
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        print(f"✗ Zip file not found: {zip_path}")
        return False
    
    if skip_if_extracted and list(extract_to.glob("*")):
        print(f"✓ Archive already extracted to: {extract_to}")
        return True
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📦 Extracting {zip_path.name} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extraction complete.")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

    print("📁 Rearranging folders for DynamicWorld (e.g., EH/WH hemispheres)...")
    # Add your specific folder reorganization code here based on the EH/WH structure.
    
    return True
