from PIL import Image
import os

folder_names = ['car_gtr_formula', 'car_gtr_formula_v2', 'car_truck_gtr', 'formula_03_04', 'formula_v1', 'creature_pit_v1_coarse', 'creature_pit_v2_coarse']
for folder_name in folder_names:
    mask_dir = f"/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/{folder_name}"
    mask_files = [f for f in os.listdir(mask_dir) if f.startswith("mask_") and f.endswith(".png")]
    for fname in mask_files:
        path = os.path.join(mask_dir, fname)
        img = Image.open(path).convert("L")  # Convert to grayscale
        img.save(path)
        print(f"Converted {fname} to single-channel")

# mask_dir = "/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/car_classic_hatchback"  # e.g., "dataset/category/chair_09_299"
# mask_files = [f for f in os.listdir(mask_dir) if f.startswith("mask_") and f.endswith(".png")]

# for fname in mask_files:
#     path = os.path.join(mask_dir, fname)
#     img = Image.open(path).convert("L")  # Convert to grayscale
#     img.save(path)
#     print(f"Converted {fname} to single-channel")