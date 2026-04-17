import zipfile
from pathlib import Path

def extract_zip(zip_path: Path, extract_to: Path, skip_if_extracted: bool = True) -> None:
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")
    
    if skip_if_extracted and list(extract_to.glob("*")):
        print(f"Archive {zip_path.name} already extracted to: {extract_to}, skipping extraction.")
    
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path.name} to {extract_to}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print("Extraction complete.")

def cleanup_zip(zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
        print(f"Deleted zip file: {zip_path}")
    else:
        print(f"Zip file not found for cleanup: {zip_path}")


def extract_and_rearrange_dw(zip_path: Path, extract_to: Path, skip_if_extracted: bool = True) -> None:

    expert_zip = extract_to / "Experts_tiles.zip"
    non_expert_zip = extract_to / "Non_expert_tiles.zip"
    validation_zip = extract_to / "validation_set_tiles.zip"
    
    try:
        extract_zip(zip_path, extract_to, skip_if_extracted)
        extract_zip(expert_zip, extract_to / "Experts", skip_if_extracted)
        extract_zip(non_expert_zip, extract_to / "Non_Experts", skip_if_extracted)
        extract_zip(validation_zip, extract_to / "Validation", skip_if_extracted)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # cleanup_zip(zip_path)
        cleanup_zip(expert_zip)
        cleanup_zip(non_expert_zip)
        cleanup_zip(validation_zip)

if __name__ == "__main__":
    # Example usage
    zip_file = Path("data/dynamicworld.zip")
    extract_folder = Path("data/dynamicworld_extracted")
    extract_and_rearrange_dw(zip_path=zip_file, extract_to=extract_folder)