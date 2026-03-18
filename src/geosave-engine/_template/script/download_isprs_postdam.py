import os
import subprocess
import urllib.request
from pathlib import Path
from tqdm import tqdm
import yaml

class DownloadProgressBar(tqdm):
    """Progress bar for urllib download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_with_progress(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def check_7z():
    """Check if 7z is installed."""
    try:
        subprocess.run(["7z"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def main():
    # Check if 7z is installed
    if not check_7z():
        print("Error: 7z is not installed.")
        print("Please install 7-Zip:")
        print("  - Linux: sudo apt install p7zip-full")
        print("  - macOS: brew install p7zip")
        print("  - Windows: Download from https://www.7-zip.org/")
        return
    
    url = "https://www.kaggle.com/api/v1/datasets/download/deasadiqbal/private-data-1"
    zip_path = "./dataset/isprs_postdam/ISPRS-Postdam.zip"
    extract_dir = "./dataset/isprs_postdam/"

    os.makedirs(extract_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        download_with_progress(url, zip_path)
        print("Download complete.")
    else:
        print("Zip file already exists. Skipping download.")

    # Create image and label directories
    image_dir = Path(extract_dir) / "images"
    label_dir = Path(extract_dir) / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract Images directly
    print("Extracting Images to image/...")
    subprocess.run(["7z", "e", "-aos", zip_path, "patches/Images/*", f"-o{image_dir}"], check=True)
    print(f"Images extracted to {image_dir}")
    
    # Extract Labels directly
    print("Extracting Labels to label/...")
    subprocess.run(["7z", "e", "-aos", zip_path, "patches/Labels/*", f"-o{label_dir}"], check=True)
    print(f"Labels extracted to {label_dir}")
    
    print("Extraction complete.")
    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")

if __name__ == "__main__":
    main()



    