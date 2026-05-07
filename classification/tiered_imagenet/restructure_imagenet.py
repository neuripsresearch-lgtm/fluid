import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
# Path relative to your python_scripts folder based on your screenshot
SOURCE_ROOT = Path("./data/tiered_imagenet")
DEST_ROOT = Path("./data/tiered_imagenet_standard")

# Split ratios (Must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Set to True to move files (faster, saves space, destroys original). 
# Set to False to copy files (safer, keeps original).
MOVE_FILES = False 

# =================================================

def get_all_classes(source_root):
    """
    Scans the existing train/test/val folders to find all unique class directories.
    Returns a dict mapping class_name -> current_path
    """
    class_map = {}
    existing_splits = ['train', 'test', 'val']
    
    print("Scanning existing dataset structure...")
    for split in existing_splits:
        split_path = source_root / split
        if not split_path.exists():
            continue
            
        # Iterate over class folders (e.g., n01440764)
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                # Since classes are currently mutually exclusive, 
                # we just record where each class currently lives.
                class_map[class_dir.name] = class_dir
                
    return class_map

def restructure_dataset():
    if not SOURCE_ROOT.exists():
        print(f"Error: Source directory {SOURCE_ROOT} does not exist.")
        return

    # 1. Identify all classes and where they are currently located
    class_map = get_all_classes(SOURCE_ROOT)
    all_classes = list(class_map.keys())
    
    print(f"Found {len(all_classes)} unique classes.")
    
    # 2. Create Destination Structure
    for split in ['train', 'val', 'test']:
        (DEST_ROOT / split).mkdir(parents=True, exist_ok=True)

    # 3. Process each class
    print("Reshuffling and distributing data...")
    
    for class_name in tqdm(all_classes):
        current_class_dir = class_map[class_name]
        
        # Get all valid image files
        images = [f for f in current_class_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # Shuffle specifically for this class
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # Remaining goes to test to ensure sum is n_total
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        # 4. Copy/Move files to new destination
        for split_name, split_images in splits.items():
            # Create class directory in destination (e.g., dest/train/n01440764)
            dest_class_dir = DEST_ROOT / split_name / class_name
            dest_class_dir.mkdir(exist_ok=True)
            
            for img_path in split_images:
                dest_file_path = dest_class_dir / img_path.name
                
                if MOVE_FILES:
                    shutil.move(str(img_path), str(dest_file_path))
                else:
                    shutil.copy2(str(img_path), str(dest_file_path))

    print("\n-------------------------------------------")
    print("Processing Complete!")
    print(f"New dataset created at: {DEST_ROOT.resolve()}")
    if not MOVE_FILES:
        print("Note: Original data was preserved. If satisfied, you can delete the old folder.")

if __name__ == "__main__":
    restructure_dataset()