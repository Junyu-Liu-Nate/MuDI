import os
from PIL import Image

def upscale_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            with Image.open(img_path) as img:
                upscaled_img = img.resize((1024, 1024), Image.BICUBIC)
                upscaled_img.save(output_path)

if __name__ == "__main__":
    folder_names = ['car_gtr_formula', 'car_gtr_formula_v2', 'car_truck_gtr', 'formula_03_04', 'formula_v1', 'creature_pit_v1_coarse', 'creature_pit_v2_coarse']
    for folder_name in folder_names:
        in_folder = f"/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/{folder_name}"
        out_folder = f"/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/{folder_name}"
        upscale_images(in_folder, out_folder)
    
    # in_folder = "/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/car_classic_hatchback"
    # out_folder = "/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/car_classic_hatchback"
    # upscale_images(in_folder, out_folder)