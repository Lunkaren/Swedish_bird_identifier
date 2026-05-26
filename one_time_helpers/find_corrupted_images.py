import os
from pathlib import Path
from PIL import Image

data_dir = "artportalen_images"
removed_count = 0

print("Scanning dataset for empty or corrupted images...")

for filepath in Path(data_dir).rglob("*"):
    if filepath.is_file():
        # 1. Check for 0-byte files
        if os.path.getsize(filepath) == 0:
            print(f"Removed empty file: {filepath}")
            os.remove(filepath)
            removed_count += 1
            continue

        # 2. Check for unreadable/corrupted files
        try:
            with Image.open(filepath) as img:
                img.verify()  # Validates the image header without loading the full array
        except Exception as e:
            print(f"Removed corrupted file: {filepath} | Error: {e}")
            os.remove(filepath)
            removed_count += 1

print(f"\nCleanup complete. Removed {removed_count} bad files.")