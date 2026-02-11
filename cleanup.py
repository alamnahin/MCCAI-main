import os
import pandas as pd
from pathlib import Path

def cleanup_extra_images(csv_path, images_dir, dry_run=True):
   
    try:
        df = pd.read_csv(csv_path)
        valid_images = set(df['Image'].str.strip())
        print(f"Loaded {len(valid_images)} valid image filenames from {csv_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return

    files_checked = 0
    files_to_delete = []

    print(f"\nScanning '{images_dir}'...")
    
    for file_path in images_path.iterdir():
        if file_path.is_file():
            files_checked += 1
            if file_path.name not in valid_images:
                files_to_delete.append(file_path)

    print(f"Scan complete. Found {len(files_to_delete)} extra files out of {files_checked} total files.\n")

    if not files_to_delete:
        print("No extra files found. The directory is clean.")
        return

    if dry_run:
        print("--- DRY RUN MODE (No files deleted) ---")
        print("The following files would be deleted:")
        for f in files_to_delete[:10]:
            print(f"  [WOULD DELETE] {f.name}")
        if len(files_to_delete) > 10:
            print(f"  ... and {len(files_to_delete) - 10} more.")
        print("\nTo actually delete these files, set dry_run=False in the script.")
    else:
        print("--- DELETING FILES ---")
        deleted_count = 0
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Deleted: {f.name}") 
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {f.name}: {e}")
        
        print(f"\nSuccessfully deleted {deleted_count} files.")

if __name__ == "__main__":
    CSV_FILE = './data/miccai_final_dataset.csv'
    IMAGES_FOLDER = './data/images'
    
    DRY_RUN = False 
    
    cleanup_extra_images(CSV_FILE, IMAGES_FOLDER, dry_run=DRY_RUN)