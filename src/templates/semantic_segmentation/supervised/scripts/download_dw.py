import gdown
from pathlib import Path

def download_dynamicworld(gdrive_url: str, output_path: Path, skip_if_exists: bool = True) -> bool:
    if output_path.exists():
        if skip_if_exists:
            print(f"DynamicWorld zip already exists: {output_path}")
            return True
        else:
            output_path.unlink()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading DynamicWorld to {output_path}...")
        gdown.download(gdrive_url, str(output_path), quiet=False)
        print(f"Download complete: {output_path}")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    gdrive_url = "https://drive.google.com/file/d/1A07lPCzsWcm9PB8JUJRWKgyfH4H6tABV/view?usp=sharing"
    output_zip = Path("data/dynamicworld.zip")
    download_dynamicworld(gdrive_url=gdrive_url, output_path=output_zip)