from PIL import Image
import os

mask_dir = "/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/chair_09_299"  # e.g., "dataset/category/chair_09_299"
mask_files = [f for f in os.listdir(mask_dir) if f.startswith("mask_") and f.endswith(".png")]

for fname in mask_files:
    path = os.path.join(mask_dir, fname)
    img = Image.open(path).convert("L")  # Convert to grayscale
    img.save(path)
    print(f"Converted {fname} to single-channel")