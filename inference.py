import os
import argparse
from PIL import Image
import json
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import random
import torchvision.transforms.functional as F
from functools import partial

def bbox_mask(mask_image):
    mask = np.array(mask_image)
    threshold_value = 128
    mask_binary = (mask > threshold_value).astype(np.uint8) * 255
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (xmin, ymin, xmax, ymax)

def image_process(image, mask, margin=32, fixed_scale=None, size_factor=0.5, out_size=(128,128), num_subject=3):
    original_size = image.size[0]
    bbox = bbox_mask(mask)
    height, width = bbox[3] - bbox[1], bbox[2] - bbox[0]
    width_per_object = out_size[1] // num_subject
    width_per_object -= width_per_object % 8
    if fixed_scale is not None:
        rescaling_factor = min((width_per_object-margin)/width, (out_size[0]-margin)/height)*fixed_scale
    else:
        max_factor = min((width_per_object-margin)/width, (out_size[0]-margin)/height)
        rescaling_factor = np.random.uniform(max_factor*size_factor, max_factor)
    new_size = int(original_size * rescaling_factor)
    image = image.resize((new_size, new_size))
    mask = mask.resize((new_size, new_size))
    return image, mask

def get_start(seq, margin, width, img_w, num_subjects):
    if seq == 0:
        return margin
    elif seq == num_subjects -1:
        return width - margin - img_w
    else:
        return (width // (num_subjects*2)) * (2*seq +1) - (img_w //2)

def image_collage(images_, image_map_size=(3,1024,1024), margin=10, height_sync=False, device='cpu', dtype=torch.float32):
    num_subjects = len(images_)
    idxs = random.sample(range(num_subjects), num_subjects)
    imgs, masks = [], []
    for idx in idxs:
        pil_img, pil_mask = images_[idx]['image'], images_[idx]['mask']
        tensor_img = F.to_tensor(pil_img).to(device=device, dtype=dtype)
        tensor_mask = F.to_tensor(pil_mask).to(device=device, dtype=dtype)
        masks.append(tensor_mask)
        imgs.append(tensor_img)
    _, H, W = image_map_size
    image_map = torch.zeros((3,H,W), device=device, dtype=dtype)
    mask_map = torch.zeros((H,W), device=device, dtype=dtype)
    y_positions = []
    if height_sync:
        max_h = max(img.size(1) for img in imgs)
        y_sync = random.randint(0, H - max_h) if H > max_h else 0
        y_positions = [y_sync]*num_subjects
    else:
        for img in imgs:
            img_h = img.size(1)
            y_positions.append(random.randint(0, max(0, H - img_h)))
    x_margins = [random.randint(1,margin)] + [random.randint(-margin,margin) for _ in range(num_subjects-2)] + [random.randint(1,margin)]
    sequence = random.sample(range(num_subjects), num_subjects)
    for seq in sequence:
        img, mask = imgs[seq], masks[seq]
        y_pos, x_margin = y_positions[seq], x_margins[seq]
        img_h, img_w = img.size(1), img.size(2)
        x_start = get_start(seq, x_margin, W, img_w, num_subjects)
        mask_squeezed = mask if mask.dim()==2 else mask[0]
        mask_map[y_pos:y_pos+img_h, x_start:x_start+img_w] += mask_squeezed
        union_mask = (mask_map>1.5)
        image_map[:, union_mask] = 0
        image_map[:, y_pos:y_pos+img_h, x_start:x_start+img_w] += img * mask
        mask_map = (mask_map>0).to(dtype=dtype)
    image_map[:, mask_map==0] = 255
    return F.to_pil_image(image_map.clamp(0,1).cpu()), F.to_pil_image(mask_map.unsqueeze(0).clamp(0,1).cpu())

def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
    if denoising_start is None:
        init_timestep = min(int(num_inference_steps*strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep,0)
    else:
        t_start =0
    timesteps = self.scheduler.timesteps[t_start*self.scheduler.order:]
    if denoising_start is not None:
        discrete_cutoff = int(round(self.scheduler.config.num_train_timesteps - (denoising_start*self.scheduler.config.num_train_timesteps)))
        num_inference_steps = (timesteps<discrete_cutoff).sum().item()
        if self.scheduler.order==2 and num_inference_steps%2==0:
            num_inference_steps+=1
        timesteps = timesteps[-num_inference_steps:]
        return timesteps, num_inference_steps
    return timesteps, num_inference_steps -t_start

def add_noise(pipe, latents, strength=1.0, noise=None):
    if not hasattr(pipe,'get_timesteps'):
        pipe.get_timesteps = partial(get_timesteps, pipe)
    num_inference_steps=50
    device = latents.device
    noise = randn_tensor(latents.shape, device=device, dtype=latents.dtype) if noise is None else noise
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps,_= pipe.get_timesteps(num_inference_steps, strength, device, denoising_start=None)
    latent_timestep = timesteps[:1]
    return pipe.scheduler.add_noise(latents, noise, latent_timestep)

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        lines=f.readlines()
    id_map, entries={},[]
    for line in lines:
        obj=json.loads(line)
        if 'id' in obj and isinstance(obj['id'], dict):
            id_map=obj['id']
        else:
            entries.append(obj)
    return id_map, entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, nargs='+', required=True, help='Multiple prompts')
    parser.add_argument('--instance_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--vae_id', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae_id, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id, vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
    pipe = pipe.to(args.device)
    pipe.load_lora_weights(args.lora_path)
    inference_device = pipe.device

    # Prepare LoRA inference subfolder
    lora_folder = os.path.dirname(args.lora_path)
    inference_folder = os.path.join(lora_folder, 'inference')
    os.makedirs(inference_folder, exist_ok=True)

    id_map, entries = load_metadata(os.path.join(args.instance_dir, 'metadata.jsonl'))

    for prompt in args.prompts:
        prompt_tokens = [tok for tok in id_map.values() if tok in prompt]
        images_ = []
        for ent in entries:
            if id_map[ent['id']] in prompt_tokens:
                img_path = os.path.join(args.instance_dir, ent['file_name'])
                mask_path = os.path.join(args.instance_dir, ent['mask_path'])
                image = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path)
                image, mask = image_process(image, mask, margin=16, fixed_scale=0.8, size_factor=1, out_size=(768, 1344), num_subject=len(prompt_tokens))
                images_.append({'image': image, 'mask': mask})

        bg, mask = image_collage(images_, (3, 1024, 1024), margin=32, height_sync=False, device=inference_device, dtype=torch.float16)
        mask_ = mask.resize((mask.size[0] // 8, mask.size[1] // 8))
        mask_ = np.array(mask_) > 0
        mask_binary = torch.tensor(mask_, device=inference_device)

        with torch.no_grad():
            latent = pipe.vae.encode(pipe.image_processor.preprocess(bg).to(inference_device, torch.float16)).latent_dist.sample()
        init_latent = latent * mask_binary.to(latent.dtype)

        gamma = 3

        # Generate 10 samples per prompt
        for i in range(10):
            generator = torch.manual_seed(i)
            noise = randn_tensor(init_latent.shape, device=inference_device, dtype=torch.float16, generator=generator)
            init_latent_ = add_noise(pipe, pipe.vae.config.scaling_factor * init_latent * gamma, strength=1., noise=noise)
            init_latent_ /= pipe.scheduler.init_noise_sigma

            img = pipe(prompt=prompt, negative_prompt='sticker, collage, cartoon, abstract glitch, blurry',
                       latents=init_latent_, cross_attention_kwargs={'scale': 1}).images[0]

            prompt_slug = prompt.replace(' ', '_').replace(',', '').replace('.', '')[:100]
            save_path = os.path.join(inference_folder, f'{prompt_slug}_sample_{i}.jpg')
            img.save(save_path)
            print(f'Saved result to {save_path}')

if __name__=='__main__':
    main()