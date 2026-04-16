import gdown
from pathlib import Path

def download_dynamicworld(gdrive_url: str, output_path: str, skip_if_exists: bool = True) -> bool:
    output_pathlib = Path(output_path)
    
    if output_pathlib.exists():
        if skip_if_exists:
            print(f"DynamicWorld zip already exists: {output_pathlib}")
            return True
        else:
            output_pathlib.unlink()
    
    output_pathlib.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading DynamicWorld to {output_pathlib}...")
        gdown.download(gdrive_url, str(output_pathlib), quiet=False)
        print(f"Download complete: {output_pathlib}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    gdrive_url = "https://drive.google.com/uc?id=YOUR_DW_FILE_ID"
    output_zip = "data/dynamicworld.zip"
    download_dynamicworld(gdrive_url=gdrive_url, output_path=output_zip)