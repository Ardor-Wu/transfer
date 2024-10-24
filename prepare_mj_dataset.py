import os
import random
import shutil
from tqdm import tqdm
from PIL import Image

# Paths
source_dir = '/scratch/qilong3/transferattack/data/mjimages'
train_dir = '/scratch/qilong3/transferattack/data/midjourney/train/class'
val_dir = '/scratch/qilong3/transferattack/data/midjourney/val/class'

# Ensure the target directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of all image files in the source directory
all_files = [
    f for f in os.listdir(source_dir)
    if os.path.isfile(os.path.join(source_dir, f))
]

# Check if there are enough images
total_images = len(all_files)
if total_images < 11000:
    raise ValueError(
        f"Not enough images in source directory. Found {total_images}, need at least 11000."
    )

# Randomly sample 11,000 images
sampled_files = random.sample(all_files, 11000)

# Split into train and validation sets
train_samples = sampled_files[:10000]
val_samples = sampled_files[10000:]

# Copy files to the train directory with progress bar and sanity check
print("Copying training images...")
for filename in tqdm(train_samples):
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(train_dir, filename)
    # Check sanity of image
    try:
        with Image.open(src_path) as img:
            img.verify()  # Verify that it is an image
        # Copy the file
        shutil.copy2(src_path, dst_path)
    except (IOError, SyntaxError) as e:
        print(f"Skipping corrupted image: {filename}")

# Copy files to the validation directory with progress bar and sanity check
print("Copying validation images...")
for filename in tqdm(val_samples):
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(val_dir, filename)
    # Check sanity of image
    try:
        with Image.open(src_path) as img:
            img.verify()
        # Copy the file
        shutil.copy2(src_path, dst_path)
    except (IOError, SyntaxError) as e:
        print(f"Skipping corrupted image: {filename}")

print("Sampling complete.")
