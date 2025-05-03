#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from custom_utils.dataset import DreamBoothMultiSynthDataset
from modules.concept_predictor import ConceptClassifierSegmenter

if is_wandb_available():
    import wandb

from datetime import datetime
from collections import defaultdict

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


def save_model_card(
    repo_id: str,
    use_dora: bool,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    vae_path=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# {'SDXL' if 'playground' not in base_model else 'Playground'} LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    if "playground" in base_model:
        model_description += """\n
## License

Please adhere to the licensing terms as described [here](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++" if "playground" not in base_model else "playground-v2dot5-community",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora" if not use_dora else "dora",
        "template:sd-lora",
    ]
    if "playground" in base_model:
        tags.extend(["playground", "playground-diffusers"])
    else:
        tags.extend(["stable-diffusion-xl", "stable-diffusion-xl-diffusers"])

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


# def log_validation(
#     pipeline,
#     args,
#     accelerator,
#     pipeline_args,
#     epoch,
#     is_final_validation=False,
# ):
#     logger.info(
#         f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
#         f" {args.validation_prompt}."
#     )

#     # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
#     scheduler_args = {}

#     if not args.do_edm_style_training:
#         if "variance_type" in pipeline.scheduler.config:
#             variance_type = pipeline.scheduler.config.variance_type

#             if variance_type in ["learned", "learned_range"]:
#                 variance_type = "fixed_small"

#             scheduler_args["variance_type"] = variance_type

#         pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     # run inference
#     generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
#     # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
#     # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
#     if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
#         autocast_ctx = nullcontext()
#     else:
#         autocast_ctx = torch.autocast(accelerator.device.type)

#     with autocast_ctx:
#         images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

#     for tracker in accelerator.trackers:
#         phase_name = "test" if is_final_validation else "validation"
#         if tracker.name == "tensorboard":
#             np_images = np.stack([np.asarray(img) for img in images])
#             tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
#         if tracker.name == "wandb":
#             tracker.log(
#                 {
#                     phase_name: [
#                         wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
#                     ]
#                 }
#             )

#     del pipeline
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     return images

def log_validation(pipeline, args, accelerator, pipeline_args_base, epoch, is_final_validation=False):
    if pipeline_args_base['validation_type'] == 'ori':
        prompts = args.val_ori_prompts
    elif pipeline_args_base['validation_type'] == 'mix':
        prompts = args.val_mix_prompts
    else:
        raise ValueError(f"Unknown validation type: {pipeline_args_base['validation_type']}")

    logger.info(
        f"Running validation... Generating {args.num_validation_images} images for each prompt..."
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    all_images = []

    # Determine tracker logging name
    if is_final_validation:
        phase_name = "test_ori" if pipeline_args_base['validation_type'] == 'ori' else "test_mix"
    else:
        phase_name = "validation_ori" if pipeline_args_base['validation_type'] == 'ori' else "validation_mix"

    # For wandb, collect all images under the same key
    wandb_log_images = []

    for p_idx, prompt in enumerate(prompts):
        pipeline_args = pipeline_args_base.copy()
        pipeline_args["prompt"] = prompt

        with torch.autocast(accelerator.device.type) if not torch.backends.mps.is_available() else nullcontext():
            images = [
                pipeline(**pipeline_args, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]
        all_images.extend(images)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(f"{phase_name}/{p_idx}", np_images, epoch, dataformats="NHWC")
            elif tracker.name == "wandb":
                wandb_log_images.extend([
                    wandb.Image(image, caption=f"[{p_idx}] {prompt}") for image in images
                ])

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({phase_name: wandb_log_images})

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=("experimemt name"),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    # parser.add_argument(
    #     "--instance_data_dir",
    #     type=str,
    #     default=None,
    #     help=("A folder containing the training data. "),
    # )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    # parser.add_argument(
    #     "--instance_prompt",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    # )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    # parser.add_argument(
    #     "--validation_prompt",
    #     type=str,
    #     default=None,
    #     help="A prompt that is used during validation to verify that the model is learning.",
    # )
    parser.add_argument(
        "--val_ori_prompts",
        nargs="+",  # Accepts multiple prompts via space or quoting
        default=None,
        help="List of validation prompts used during validation - validate ori comb. of concepts.",
    )
    parser.add_argument(
        "--val_mix_prompts",
        nargs="+",  # Accepts multiple prompts via space or quoting
        default=None,
        help="List of validation prompts used during validation - validate mix comb. of concepts.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--do_edm_style_training",
        default=False,
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_kohya_format",
        action="store_true",
        help="Flag to additionally generate final state dict in the Kohya format so that it becomes compatible with A111, Comfy, Kohya, etc.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "Wether to train a DoRA as proposed in- DoRA: Weight-Decomposed Low-Rank Adaptation https://arxiv.org/abs/2402.09353. "
            "Note: to use DoRA you need to install peft from main, `pip install git+https://github.com/huggingface/peft.git`"
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save token_embeds & LoRA weight every X updates steps.",
    )
    parser.add_argument(
        "--segmix_prob",
        type=float,
        default=0.6,
        help="Unmix regularization weight",
    )
    # segmix_start_step
    parser.add_argument(
        "--segmix_start_step",
        type=int,
        default=0,
        help="when we start Seg-Mix augmentation",
    )
    # relative scale
    parser.add_argument(
        "--relative_scale",
        type=float,
        default=0.,
        help="Determine relative scale",
    )
    parser.add_argument(
        "--soft_alpha",
        type=float,
        default=1.0,
        help="weight for soft masked diffusion loss, 0 means hard masked diffusion loss, 1.0 means no mask.",
    )
    parser.add_argument(
        "--dco_beta",
        type=float,
        default=1000
    )

    ##### Newly added helper funcs
    def parse_comma_separated_list(string):
        return string.split(',')

    def parse_list_of_lists(string):
        # Split the string into chunks where each chunk is a sublist
        sublists = string.split(';')  # Use semicolon to separate sublists
        return [sublist.strip().split(',') for sublist in sublists if sublist]  # Split each sublist into items

    def parse_nested_list_of_strings(string):
        # Assuming sublists are separated by a semicolon and elements by a comma
        return [item.split(',') for item in string.split(';') if item]

    #####----------- Newly added for part-level concepts learning -----------#####
    ##########---------- Dataset ----------##########
    parser.add_argument(
        "--subject_name",
        type=str,
        default="chair",
        required=True,
        help="Subject name, like chair and racing car.",
    )
    parser.add_argument(
        "--cross_img_sample",
        action="store_true",
        help="Whether to perform cross-image sampling.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    # parser.add_argument(
    #     "--resolution",
    #     type=int,
    #     default=512,
    #     help=(
    #         "The resolution for input images, all the images in the train/validation dataset will be resized to this"
    #         " resolution"
    #     ),
    # )
    # parser.add_argument(
    #     "--center_crop",
    #     default=False,
    #     action="store_true",
    #     help=(
    #         "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
    #         " cropped. The images will be resized to the resolution first before cropping."
    #     ),
    # )
    parser.add_argument(
        "--randomize_unused_mask_areas",
        default=False,
        action="store_true",
        help=(
            "Whether to randomize the unused mask areas. Aim to further improve disentanglement"
        ),
    )
    parser.add_argument(
        "--set_bg_white",
        default=False,
        action="store_true",
        help=(
            "Whether to set the background + unused mask areas to white."
        ),
    )
    # parser.add_argument(
    #     "--dataloader_num_workers",
    #     type=int,
    #     default=0,
    #     help=(
    #         "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    #     ),
    # )
    parser.add_argument("--img_log_steps", type=int, default=200)
    # parser.add_argument("--num_of_assets", type=int, default=1)
    parser.add_argument("--assets_indices_lists", type=parse_list_of_lists, default=[], help="Input as '1,2,3;4,5,6;7,8,9' for [[1, 2, 3], [4, 5, 6], [7, 8, 9]]")
    # parser.add_argument("--initializer_tokens", type=str, nargs="+", default=[])
    parser.add_argument("--initializer_tokens_list", type=parse_nested_list_of_strings, default=[], help="Input nested lists as 'str1,str2;str3,str4'")
    # parser.add_argument("--val_mix_prompts", 
    #                     nargs='+',
    #                     default=None,
    #                     help="A list of prompts that are sampled during validation for inference.",
    # )
    parser.add_argument("--final_inference_prompts", 
                        nargs='+',
                        default=None,
                        help="A list of prompts that are sampled during final inference.",
    )

    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<asset>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--use_all_synth_imgs",
        action="store_true",
        help="Whether to use all synthetic images for training.",
    )
    # parser.add_argument(
    #     "--synth_type",
    #     type=int,
    #     default=0,
    #     help="Sythn type for the images synthesize by sampling across images.",
    # )
    parser.add_argument(
        "--sample_type",
        type=str,
        choices=["fixed-num", "random-num", "per-part", "per-subject"],
        required=False,
        help="Specify the concept combination method to use. Choices are: random, per-part, per-subject."
    )
    parser.add_argument(
        "--synth_type",
        type=str,
        choices=["4-corner", "2-subject", "random-no-overlap", "random-overlap"],
        required=False,
        help="Specify the img synthesis method to use. Choices are: 4-corner, 2-subject, random-no-overlap, random-overlap."
    )
    parser.add_argument(
        "--sythn_detailed_prompt",
        action="store_true",
        help="Whether to use detailed prompt for synthetic image.",
    )
    parser.add_argument(
        "--train_detailed_prompt",
        action="store_true",
        help="Whether to use detailed prompt for train image - the provided image.",
    )
    parser.add_argument(
        "--use_all_instance",
        action="store_true",
        help="For running BaS baseline - use all union sampled instance images.",
    )
    ##### Learn token to represent background images
    parser.add_argument(
        "--use_bg_tokens",
        action="store_true",
        help="Whether to learn bg concept.",
    )
    parser.add_argument(
        "--bg_indices",
        type=parse_comma_separated_list,
        default=[],
        help="A list of indices for the background images.",
    )
    parser.add_argument(
        "--bg_placeholder_token",
        type=str,
        default="<bg>",
        help="A token to use as a placeholder for the bg concept.",
    )
    parser.add_argument(
        "--bg_initializer_tokens",
        type=parse_comma_separated_list,
        default=[],
        help="A comma-separated list of tokens to use as initializers for the bg concepts.",
    )
    parser.add_argument(
        "--bg_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of bg images.",
    )

    ##########---------- Training ----------##########
    parser.add_argument(
        "--do_not_apply_masked_loss",
        action="store_false",
        help="Use masked loss instead of standard epsilon prediciton loss",
        dest="apply_masked_loss"
    )
    parser.add_argument(
        "--apply_bg_loss",
        action="store_true",
        help="Add background loss to the diffusion loss - enforce the background to be white",
    )
    parser.add_argument(
        "--bg_loss_weight",
        type=float,
        default=0.05,
        help="weight for the background loss",
    )
    parser.add_argument(
        "--train_concept_predictor",
        action="store_true",
        help="Use concept train_concept_predictor to predict the concept tokens - Maximize Mutual Information",
    )
    parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["classifier", "classifier_seg", "classifier_time", "classifier_seg_time", "classifier_seg_time_film", "regressor"],
        help="Specify which concept predictor to use: classifier or regressor."
    )
    parser.add_argument(
        "--concept_pred_weight",
        type=float,
        default=1.0,
        help="weight for the train_concept_predictor loss",
    )
    parser.add_argument(
        "--concept_pred_seg_scale",
        type=float,
        default=1.0,
        help="scale for the seg loss in train_concept_predictor loss. If 1.0, it's averagly 1/10 of the classification loss.",
    )
    parser.add_argument(
        "--train_token_embeds",
        action="store_true",
        help="Whether to train the token embeddings.",
    )

    #####--------------------------------------------------------------------#####

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            # raise ValueError("You must specify prompt for class images.")
            pass
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            # warnings.warn("You need not use --class_prompt without --with_prior_preservation.")
            if args.with_prior_preservation:
                warnings.warn("Class_prompt should be provided as jsonl")

    return args


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask_map"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    do_segmix = [example["do_segmix"] for example in examples]
    is_first = [example["is_first"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask_map"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = torch.stack(mask)
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts, "do_segmix": do_segmix, "is_first": is_first, "mask": mask}
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def ensure_mask_format(mask):
    # Ensure shape [B, 1, H, W]
    if mask.ndim == 3:  # [B, H, W]
        mask = mask.unsqueeze(1)
    elif mask.ndim == 5:  # [B, P, 1, H, W]
        mask = torch.max(mask, dim=1).values  # [B, 1, H, W]
    return mask

def main(args):
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d-%H-%M")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.do_edm_style_training and args.snr_gamma is not None:
        raise ValueError("Min-SNR formulation is not supported when conducting EDM-style training.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    scheduler_type = determine_scheduler_type(args.pretrained_model_name_or_path, args.revision)
    if "EDM" in scheduler_type:
        args.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        logger.info("Performing EDM-style training!")
    elif args.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    #####--------- Add and initialize specialized tokens for concepts ---------#####
    # -------- Add and initialize new tokens --------
    total_assets_indices = [idx for group in args.assets_indices_lists for idx in group]
    all_placeholder_tokens = [args.placeholder_token.replace(">", f"{idx}>") for idx in total_assets_indices]
    placeholder_tokens = [[args.placeholder_token.replace(">", f"{idx}>") for idx in group] for group in args.assets_indices_lists]
    print(f"All placeholder tokens: {all_placeholder_tokens}")

    # Add to tokenizer
    num_added_tokens = tokenizer_one.add_tokens(all_placeholder_tokens)
    tokenizer_two.add_tokens(all_placeholder_tokens)
    placeholder_token_ids = tokenizer_one.convert_tokens_to_ids(all_placeholder_tokens)

    # Resize token embeddings
    text_encoder_one.resize_token_embeddings(len(tokenizer_one))
    text_encoder_two.resize_token_embeddings(len(tokenizer_two))

    # Initialize embeddings
    token_embeds_1 = text_encoder_one.get_input_embeddings().weight.data
    token_embeds_2 = text_encoder_two.get_input_embeddings().weight.data

    if len(args.initializer_tokens_list) > 0:
        all_initializer_tokens = [t for group in args.initializer_tokens_list for t in group]
        for i, init_token in enumerate(all_initializer_tokens):
            init_id = tokenizer_one.encode(init_token, add_special_tokens=False)[0]
            token_embeds_1[placeholder_token_ids[i]] = token_embeds_1[init_id]
            token_embeds_2[placeholder_token_ids[i]] = token_embeds_2[init_id]
    else:
        # Randomly initialize from nearby
        for i in range(len(placeholder_token_ids)):
            token_embeds_1[placeholder_token_ids[i]] = token_embeds_1[-(3 * len(placeholder_token_ids)) + i]
            token_embeds_2[placeholder_token_ids[i]] = token_embeds_2[-(3 * len(placeholder_token_ids)) + i]
    #####----------------------------------------------------------------------#####

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    ###########
    # # Enable gradients only on concept token embeddings
    if args.train_token_embeds:
        embedding_1 = text_encoder_one.get_input_embeddings()
        embedding_2 = text_encoder_two.get_input_embeddings()

        # Ensure the embedding weight as a whole is marked trainable
        embedding_1.weight.requires_grad = True
        embedding_2.weight.requires_grad = True
    #############

    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    #####--------- Initialize the concept predictor ---------#####
    if args.train_concept_predictor:
        concept_predictor = ConceptClassifierSegmenter(
            latent_channels=4, latent_size=128, out_dim=len(all_placeholder_tokens), hidden_dim=256
        ).to(accelerator.device)
        concept_predictor.train()
    #####----------------------------------------------------#####

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        use_dora=args.use_dora,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            use_dora=args.use_dora,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    # Optimization parameters
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]

    if args.train_concept_predictor:
        params_to_optimize.append({"params": concept_predictor.parameters(), "lr": args.learning_rate})

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    # train_dataset = DreamBoothDataset(
    #     train_data_root=args.instance_data_dir,
    #     class_prompt=args.class_prompt,
    #     class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    #     class_num=args.num_class_images,
    #     size=args.resolution,
    #     repeats=args.repeats,
    #     center_crop=args.center_crop,
    #     image_column='file_name',
    #     caption_column='text',
    #     segmix_prob=args.segmix_prob,
    #     soft_alpha=args.soft_alpha
    # )
    train_dataset = DreamBoothMultiSynthDataset(
        instance_data_root=args.instance_data_dir,
        placeholder_tokens=placeholder_tokens,
        use_bg_tokens=args.use_bg_tokens,
        bg_data_root=args.bg_data_dir,
        bg_placeholder_tokens="",
        size=1024,
        center_crop=False,
        flip_p=0.5,
        randomize_unused_mask_areas=args.randomize_unused_mask_areas,
        set_bg_white=args.set_bg_white,
        use_all_sythn=args.use_all_synth_imgs,
        use_all_instance=args.use_all_instance,
        subject_name=args.subject_name,
        sample_type=args.sample_type,
        synth_type=args.synth_type,
        train_detailed_prompt=args.train_detailed_prompt,
        sythn_detailed_prompt=args.sythn_detailed_prompt,
    )

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=True,
    #     collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    #     num_workers=args.dataloader_num_workers,
    # )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,  # <- add this
        num_workers=4,
    )

    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    def compute_time_ids(original_size=(args.resolution, args.resolution), crops_coords_top_left=(0, 0)):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    # Handle instance prompt.
    instance_time_ids = compute_time_ids()

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        if not args.train_text_encoder:
            pass
            # class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
            #     args.class_prompt, text_encoders, tokenizers
            # )

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    add_time_ids = instance_time_ids
    if args.with_prior_preservation:
        add_time_ids = torch.cat([add_time_ids, class_time_ids], dim=0)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = (
            "segmix-lora-sd-xl"
            if "playground" not in args.pretrained_model_name_or_path
            else "segmix-lora-playground"
        )
        accelerator.init_trackers(tracker_name, config=vars(args))

        if args.report_to == "wandb":
            wandb.run.name = args.run_name + "_" + date_time_string

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            logs = {}

            if global_step == args.segmix_start_step:
                print("start segmix")
                train_dataset.start_segmix = True

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                # mask = batch["mask"]

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()

                if latents_mean is None and latents_std is None:
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                else:
                    latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                    latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                    model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                    model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # print(f"batch size: {bsz}")

                # Sample a random timestep for each image
                if not args.do_edm_style_training:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()
                else:
                    # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                    # instead of discrete timesteps, so here we sample indices to get the noise levels
                    # from `scheduler.timesteps`
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                # For EDM-style training, we first obtain the sigmas based on the continuous timesteps.
                # We then precondition the final model inputs based on these sigmas instead of the timesteps.
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                if args.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                    if "EDM" in scheduler_type:
                        inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                    else:
                        inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                #################
                # # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                # if not train_dataset.custom_instance_prompts:
                #     elems_to_repeat_text_embeds = bsz // 2 if args.with_prior_preservation else bsz
                #     elems_to_repeat_time_ids = bsz // 2 if args.with_prior_preservation else bsz
                # else:
                #     elems_to_repeat_text_embeds = 1
                #     elems_to_repeat_time_ids = bsz // 2 if args.with_prior_preservation else bsz

                # print(f'add_time_ids: {add_time_ids}')
                # print(f'elems_to_repeat_time_ids: {elems_to_repeat_time_ids}')
                # # Predict the noise residual
                # unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1)}
                #################

                #################
                # Compute correct time_ids
                original_size = (args.resolution, args.resolution)
                crop_coords = (0, 0)
                target_size = (args.resolution, args.resolution)

                # Construct [6] vector: [H_orig, W_orig, crop_top, crop_left, H_target, W_target]
                time_ids = torch.tensor([*original_size, *crop_coords, *target_size], device=model_input.device, dtype=weight_dtype)

                # Repeat to match batch size
                add_time_ids = time_ids.unsqueeze(0).repeat(bsz, 1)  # shape: [bsz, 6]
                # print(f'add_time_ids: {add_time_ids}')
                unet_added_conditions = {"time_ids": add_time_ids}
                #################
                
                # print(f'prompts: {prompts}')
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=[tokenizer_one, tokenizer_two],
                    prompt=prompts,
                    text_input_ids_list=None,
                )
                # print(f'prompt_embeds shape: {prompt_embeds.shape}')
                # print(f'pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}')
                unet_added_conditions.update(
                    # {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    {"text_embeds": pooled_prompt_embeds}
                )

                # prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                # print(f'unet_added_conditions: {unet_added_conditions}')
                prompt_embeds_input = prompt_embeds
                model_pred = unet(
                    inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                if args.dco_beta > 0.0 :
                    with torch.no_grad():
                        model_pred_dco = unet(
                            inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                            timesteps,
                            prompt_embeds_input,
                            added_cond_kwargs=unet_added_conditions,
                            cross_attention_kwargs={'scale': 0},
                            return_dict=False,
                        )[0]
                weighting = None
                if args.do_edm_style_training:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas.
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    if "EDM" in scheduler_type:
                        model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_model_input
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                noisy_model_input / (sigmas**2 + 1)
                            )
                    # We are not doing weighting here because it tends result in numerical problems.
                    # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                    # There might be other alternatives for weighting as well:
                    # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                    if "EDM" not in scheduler_type:
                        weighting = (sigmas**-2.0).float()

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = model_input if args.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        model_input
                        if args.do_edm_style_training
                        else noise_scheduler.get_velocity(model_input, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    mask, prior_mask = torch.chunk(batch["mask"], 2, dim=0)
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    if weighting is not None:
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                target_prior.shape[0], -1
                            ),
                            1,
                        )
                        prior_loss = (prior_loss * prior_mask).mean()
                    else:
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="none")
                        prior_loss = (prior_loss * prior_mask).mean()

                if args.snr_gamma is None:
                    # print(f'Check snr_gamma is None')
                    if weighting is not None:
                        print(f'Check weighting is not None')
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )
                        loss = (loss * mask).mean()
                        if args.dco_beta > 0.0:
                            loss_dco = torch.mean(
                                (weighting.float() * (model_pred_dco.float() - target.float()) ** 2).reshape(
                                    target.shape[0], -1
                                ),
                                1,
                            )
                            loss_dco = (loss_dco * mask).mean()
                            diff = loss - loss_dco
                            inside_term = -1 * args.dco_beta * diff
                            loss = -1 * torch.nn.LogSigmoid()(inside_term)
                    else:
                        # print(f'Check weighting is None')
                        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        # loss = (loss * mask).mean()
                        #####---------- Added for masks produced by dynamic data synth ----------#####
                        if args.apply_masked_loss:
                            # Split batch in half
                            model_pred_synth, model_pred_inst = torch.chunk(model_pred, 2, dim=0)
                            target_synth, target_inst = torch.chunk(target, 2, dim=0)
                            # print(f'model_pred_synth shape: {model_pred_synth.shape}')
                            # print(f'model_pred_inst shape: {model_pred_inst.shape}')

                            # Get masks and downsample
                            synth_masks = torch.max(batch["synth_masks"], dim=1).values
                            inst_masks = torch.max(batch["instance_masks"], dim=1).values
                            synth_masks = ensure_mask_format(batch["synth_masks"])
                            inst_masks = ensure_mask_format(batch["instance_masks"])
                            synth_masks = F.interpolate(synth_masks, size=(128, 128), mode="nearest").squeeze(1)
                            inst_masks = F.interpolate(inst_masks, size=(128, 128), mode="nearest").squeeze(1)
                            synth_masks = (synth_masks > 0.1).float()
                            inst_masks = (inst_masks > 0.1).float()

                            if args.apply_bg_loss:
                                synth_weights = synth_masks + (1.0 - synth_masks) * args.bg_loss_weight
                                inst_weights = inst_masks + (1.0 - inst_masks) * args.bg_loss_weight
                            else:
                                synth_weights = synth_masks
                                inst_weights = inst_masks

                            synth_weights = synth_weights.unsqueeze(1)
                            inst_weights = inst_weights.unsqueeze(1)

                            mse_synth = (model_pred_synth - target_synth) ** 2 * synth_weights
                            mse_inst = (model_pred_inst - target_inst) ** 2 * inst_weights
                            loss = mse_synth.mean() + mse_inst.mean()
                        else:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = (loss * mask).mean()
                        #####------------------------------------------------------------#####
                        
                        if args.dco_beta > 0.0:
                            # loss_dco = F.mse_loss(model_pred_dco.float(), target.float(), reduction="none")
                            # loss_dco = (loss_dco * mask).mean()
                            #####---------- Added for masks produced by dynamic data synth ----------#####
                            if args.apply_masked_loss:
                                # Split DCO preds and targets
                                model_pred_dco_synth, model_pred_dco_inst = torch.chunk(model_pred_dco, 2, dim=0)

                                # Downsample and binarize masks
                                synth_masks = torch.max(batch["synth_masks"], dim=1).values
                                inst_masks = torch.max(batch["instance_masks"], dim=1).values
                                # Apply fix
                                synth_masks = ensure_mask_format(batch["synth_masks"])
                                inst_masks = ensure_mask_format(batch["instance_masks"])
                                synth_masks = F.interpolate(synth_masks, size=(128, 128), mode="nearest").squeeze(1)
                                inst_masks = F.interpolate(inst_masks, size=(128, 128), mode="nearest").squeeze(1)
                                synth_masks = (synth_masks > 0.1).float()
                                inst_masks = (inst_masks > 0.1).float()

                                if args.apply_bg_loss:
                                    synth_weights = synth_masks + (1.0 - synth_masks) * args.bg_loss_weight
                                    inst_weights = inst_masks + (1.0 - inst_masks) * args.bg_loss_weight
                                else:
                                    synth_weights = synth_masks
                                    inst_weights = inst_masks

                                synth_weights = synth_weights.unsqueeze(1)
                                inst_weights = inst_weights.unsqueeze(1)

                                # Compute masked MSE for DCO
                                mse_dco_synth = (model_pred_dco_synth - target_synth) ** 2 * synth_weights
                                mse_dco_inst = (model_pred_dco_inst - target_inst) ** 2 * inst_weights
                                loss_dco = mse_dco_synth.mean() + mse_dco_inst.mean()
                            else:
                                loss_dco = F.mse_loss(model_pred_dco.float(), target.float(), reduction="none")
                                loss_dco = (loss_dco * mask).mean()
                            #####------------------------------------------------------------#####

                            diff = loss - loss_dco
                            inside_term = -1 * args.dco_beta * diff
                            loss = -1 * torch.nn.LogSigmoid()(inside_term)
                else:
                    raise NotImplementedError("SNR-based loss weights are not yet supported.")
                logs["mask_diff_loss"] = loss.detach().item()

                #####---------- Added concept predictor ----------#####
                if args.train_concept_predictor:
                    # === Get denoised latents ===
                    if noise_scheduler.config.prediction_type == "epsilon":
                        alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)  # [B,1,1,1]
                        denoised_latents = (noisy_model_input - (1 - alpha_t).sqrt() * model_pred) / alpha_t.sqrt()
                    else:
                        raise NotImplementedError("Only epsilon prediction supported for concept prediction.")

                    # === Split into synth/instance ===
                    denoised_synth, denoised_inst = torch.chunk(denoised_latents, 2, dim=0)
                    synth_token_ids = torch.stack(batch["synth_token_ids"])
                    inst_token_ids = batch["token_ids"]
                    token_id_list = [synth_token_ids, inst_token_ids]

                    synth_masks = batch["synth_masks"]
                    inst_masks = batch["instance_masks"]

                    if synth_masks.ndim == 5:
                        synth_masks = synth_masks.squeeze(2)
                    if inst_masks.ndim == 5:
                        inst_masks = inst_masks.squeeze(2)

                    synth_masks = F.interpolate(synth_masks, size=(128, 128), mode="nearest")
                    inst_masks = F.interpolate(inst_masks, size=(128, 128), mode="nearest")
                    synth_masks = (synth_masks > 0.1).float()
                    inst_masks = (inst_masks > 0.1).float()
                    mask_list = [synth_masks, inst_masks]

                    concept_pred_loss = 0.0
                    for i, (latents_i, token_ids_i, masks_i) in enumerate(zip([denoised_synth, denoised_inst], token_id_list, mask_list)):
                        # print(f"latents_i shape: {latents_i.shape}")
                        logits_cls, logits_mask = concept_predictor(latents_i)  # (1, C), (1, C, 64, 64)

                        cls_labels = torch.zeros_like(logits_cls)
                        for tid in token_ids_i[0]:
                            cls_labels[0, tid] = 1.0

                        cls_loss = F.binary_cross_entropy_with_logits(logits_cls, cls_labels)

                        # Build GT segmentation mask
                        gt_mask = torch.zeros_like(logits_mask)
                        for ch, tid in enumerate(token_ids_i[0]):
                            gt_mask[0, tid] = masks_i[0, ch]

                        seg_loss = F.binary_cross_entropy_with_logits(logits_mask, gt_mask)
                        concept_pred_loss += cls_loss + args.concept_pred_seg_scale * seg_loss
                    
                    # Scale and add to total loss
                    concept_pred_loss = args.concept_pred_weight * concept_pred_loss
                    loss = loss + concept_pred_loss
                    logs["info_loss"] = concept_pred_loss.detach().item()
                #####------------------------------------------------------------#####

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)

                if args.train_token_embeds:
                    with torch.no_grad():
                        grad_1 = embedding_1.weight.grad
                        grad_2 = embedding_2.weight.grad
                        mask_1 = torch.zeros_like(grad_1)
                        mask_2 = torch.zeros_like(grad_2)
                        for token_id in placeholder_token_ids:
                            mask_1[token_id] = 1
                            mask_2[token_id] = 1
                        grad_1 *= mask_1
                        grad_2 *= mask_2

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
                        if args.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                ###TODO: Need to check whether the embs are updated - have gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.save_steps == 0 and global_step > args.segmix_start_step-1:
                        save_path = os.path.join(args.output_dir, f"save-{global_step}")
                        if args.train_text_encoder:
                            text_encoder_one = unwrap_model(text_encoder_one)
                            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
                            )
                            text_encoder_two = unwrap_model(text_encoder_two)
                            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(text_encoder_two.to(torch.float32))
                            )
                        else:
                            text_encoder_lora_layers = None
                            text_encoder_2_lora_layers = None
                            
                        unet_lora_layers = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrap_model(unet)))
                        unet_lora_layers = {k: v.to(torch.float32) for k, v in unet_lora_layers.items()}
                        StableDiffusionXLPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_layers,
                            text_encoder_lora_layers=text_encoder_lora_layers,
                            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                        )
                        if args.output_kohya_format:
                            lora_state_dict = load_file(f"{save_path}/pytorch_lora_weights.safetensors")
                            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
                            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
                            save_file(kohya_state_dict, f"{save_path}/pytorch_lora_weights_kohya.safetensors")

                        del unet_lora_layers
                        torch.cuda.empty_cache()
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs["loss"] = loss.detach().item()
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            # if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
            if args.val_ori_prompts and args.val_mix_prompts and epoch % args.validation_epochs == 0:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    unet=accelerator.unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                # pipeline_args = {"prompt": args.validation_prompt}

                # images = log_validation(
                #     pipeline,
                #     args,
                #     accelerator,
                #     pipeline_args,
                #     epoch,
                # )

                pipeline_args = {"prompt": args.val_ori_prompts, "validation_type": 'ori', "num_inference_steps": 25}
                images = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    pipeline_args,
                    epoch,
                )

                pipeline_args = {"prompt": args.val_mix_prompts, "validation_type": 'mix', "num_inference_steps": 25}
                images = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    pipeline_args,
                    epoch,
                )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            )
            text_encoder_two = unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two.to(torch.float32))
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        if args.output_kohya_format:
            lora_state_dict = load_file(f"{args.output_dir}/pytorch_lora_weights.safetensors")
            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            save_file(kohya_state_dict, f"{args.output_dir}/pytorch_lora_weights_kohya.safetensors")

        # Final inference
        # Load previous pipeline
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        # if args.validation_prompt and args.num_validation_images > 0:
        #     pipeline_args = {"prompt": args.validation_prompt, "num_inference_steps": 25}
        #     images = log_validation(
        #         pipeline,
        #         args,
        #         accelerator,
        #         pipeline_args,
        #         epoch,
        #         is_final_validation=True,
        #     )
        if args.val_ori_prompts and args.val_mix_prompts and args.num_validation_images > 0:
            # pipeline_args = {"prompt": args.validation_prompt, "num_inference_steps": 25}
            pipeline_args = {"prompt": args.val_ori_prompts, "validation_type": 'ori', "num_inference_steps": 25}
            images = log_validation(
                pipeline,
                args,
                accelerator,
                pipeline_args,
                epoch,
            )
            
            pipeline_args = {"prompt": args.val_mix_prompts, "validation_type": 'mix', "num_inference_steps": 25}
            images = log_validation(
                pipeline,
                args,
                accelerator,
                pipeline_args,
                epoch,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                use_dora=args.use_dora,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
