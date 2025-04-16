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
    in_folder = "/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/chair_09_299"
    out_folder = "/users/ljunyu/data/ljunyu/projects/few_shot_concept/code/MuDI/dataset/category/chair_09_299"
    upscale_images(in_folder, out_folder)