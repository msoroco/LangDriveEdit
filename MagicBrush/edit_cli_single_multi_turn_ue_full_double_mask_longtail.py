from __future__ import annotations

import math
import os.path
import random
import shutil
import sys
from argparse import ArgumentParser
from tqdm import tqdm
import glob

import einops
# import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import json
import os
from diffusers import StableDiffusion3InstructPix2PixPipeline


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)



def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(ckpt, torch_dtype=torch.float16)
    return pipe



def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", type=int, nargs=2, default=[600, 366], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)  # default 2.1
    parser.add_argument("--ckpt", default="checkpoints/MagicBrush-epoch-52-step-4999.ckpt", type=str)  # default 2.1
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input_path", required=True, type=str, default='./dev/images')
    parser.add_argument("--output_path", required=True, type=str, default='./DevMagicBrushOutput')
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_iter", action="store_true")
    parser.add_argument("--use_mask", action="store_true") # eval with editing region as input
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if isinstance(args.resolution, int):
        # width, height.
        args.resolution = (args.resolution, args.resolution)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pipe = load_model_from_config(args.ckpt)
    pipe = pipe.to("cuda")

    seed = random.randint(0, 100000) if args.seed is None else args.seed

    # Dictionary of prompts to apply to each image
    # prompts_dict = {
    #     "sunny": "Change the weather conditions to sunny, making the road surface light grey and clear of precipitation.",
    #     "night": "Change the weather to become cloudy instead of clear.",
    #     "rain": "Change the weather conditions to rainy, making the road surface dark grey and wet.",
    #     "snow": "Change the weather conditions to snowy, making the road surface white and covered in snow.",
    #     "snow2": "Add snow to this scene",
    #     "morning": "Change the time of day to morning, making the sky light blue and the sun low on the horizon.",
    #     "afternoon": "Change the time of day to afternoon, making the sky light blue and the sun high in the sky.",
    #     "spring": "Change the season to spring, making the trees green and the flowers blooming.",
    #     "fall": "Change the season to fall, making the trees orange and the leaves falling.",
    #     "winter": "Change the season to winter, making the trees bare and the ground covered in snow.",
    # }
    prompts_dict = {
        "sunny2": "Change the weather conditions to sunny, removing any clouds and making the road surface dry.",
        "cloudy": "Change the weather conditions to cloudy, making the road surface dlightly darker and overcast.",
        "rain2": "Change the weather conditions to light rain, making the road surface slightly darker and wet.",
        "morning": "Change the time of day to morning, making the sky light blue and the sun low on the horizon.",
        "afternoon2": "Change the time of day to afternoon with no clouds and the sun high in the sky.",
    }


    def fit_to_resolution(image, resolution):
        target_width, target_height = resolution
        return ImageOps.fit(image, (target_width, target_height), method=Image.Resampling.LANCZOS)

    def edit_image(input_path, output_path, edit_instruction, mask_img=None, add_mask_img=None):
        torch.manual_seed(seed)
        input_image = Image.open(input_path).convert("RGB")
        width, height = input_image.size  
        input_image = fit_to_resolution(input_image, args.resolution)
        if mask_img is not None:
            mask_img = Image.open(mask_img).convert("RGB")
        else:
            # Create a blank white image with same dimensions as input
            mask_img = Image.new("RGB", (width, height), (255, 255, 255))
        mask_img = fit_to_resolution(mask_img, args.resolution)
        if add_mask_img is not None:
            add_mask_img = Image.open(add_mask_img).convert("RGB")
        else:
            # Create a blank white image with same dimensions as input
            add_mask_img = Image.new("RGB", (width, height), (255, 255, 255))
        add_mask_img = fit_to_resolution(add_mask_img, args.resolution)
        

        edited_image = pipe(
            edit_instruction,
            image=input_image,
            mask_img=mask_img,
            add_mask_img=add_mask_img,
            width=args.resolution[0],
            height=args.resolution[1],
            negative_prompt="",
            num_inference_steps=args.steps,
            image_guidance_scale=args.cfg_image,
            guidance_scale=args.cfg_text,
        ).images[0]

        edited_image.save(output_path)
        print(f"Saved edited image to {output_path}")

   
    # Get all image files from the input directory
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.avif']
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_path, f'*{ext}')))
    
    print(f"Found {len(image_files)} images to process")


    # Process each image with each prompt
    for img_path in tqdm(image_files, desc="Processing images"):
        img_basename = os.path.basename(img_path)
        img_name = os.path.splitext(img_basename)[0]
        
        for prompt_name, prompt in prompts_dict.items():
            output_filename = f"{img_name}_{prompt_name}.png"
            output_path = os.path.join(args.output_path, output_filename)
            
            if os.path.exists(output_path):
                print(f"Output file {output_path} already exists, skipping")
                continue
                
            print(f"\nProcessing {img_path} with prompt: {prompt}")
            edit_image(img_path, output_path, prompt)



if __name__ == "__main__":
    main()
