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
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import json
import os
sys.path.append("/net/acadia8a/data/msoroco/code/projects/instruct/instruct-pix2pix/stable_diffusion")

# from stable_diffusion.ldm.util import instantiate_from_config
from ldm.util import instantiate_from_config
# from safetensors.torch import load_file
# from diffusers import StableDiffusion3InstructPix2PixPipeline

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


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    print(ckpt)
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_s tage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 320], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)  # default 2.1
    parser.add_argument("--ckpt", default="checkpoints/MagicBrush-epoch-52-step-4999.ckpt", type=str)  # default 2.1
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input_path", required=True, type=str, default='./dev/images')
    parser.add_argument("--output_path", required=True, type=str, default='./DevMagicBrushOutput')
    parser.add_argument("--use_mask", action="store_true") # eval with editing region as input
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_iter", action="store_true")

    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if isinstance(args.resolution, int):
        # width, height.
        args.resolution = (args.resolution, args.resolution)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
   
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed

    # Dictionary of prompts to apply to each image
    prompts_dict = {
        "sunny": "Change the weather conditions to sunny, removing any clouds and making the road surface dry.",
        "cloudy": "Change the weather conditions to cloudy, making the road surface dlightly darker and overcast.",
        "rain": "Change the weather conditions to rainy, making the road surface darker and wet.",
        "snow": "Change the weather conditions to snowy, making the road surface white and covered in snow.",
        "snow2": "Add snow to this scene",
        "morning": "Change the time of day to morning, making the sky light blue and the sun low on the horizon.",
        "afternoon": "Change the time of day to afternoon, making the sky light blue and the sun high in the sky.",
        "spring": "Change the season to spring, making the trees green and the flowers blooming.",
        "fall": "Change the season to fall, making the trees orange and the leaves falling.",
        "winter": "Change the season to winter, making the trees bare and the ground covered in snow.",
    }

    # def fit_to_resolution(image, resolution):
    #     if image is None:
    #         return None
    #     target_width, target_height = resolution
    #     width, height = image.size
    #     factor = min(target_width / width, target_height / height)
    #     width = int((width * factor) // 64) * 64
    #     height = int((height * factor) // 64) * 64
    #     return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    
    def fit_to_resolution(image, resolution):
        if image is None:
            return None
        target_width, target_height = resolution
        # This will resize to exactly the requested dimensions
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


    def edit_image(input_path, output_path, edict_instruction, mask_img=None, add_mask_img=None):
        input_image = Image.open(input_path).convert("RGB")
        input_image = fit_to_resolution(input_image, args.resolution)

        mask_img = Image.open(mask_img).convert("RGB") if mask_img is not None else None
        mask_img = fit_to_resolution(mask_img, args.resolution)
        add_mask_image = Image.open(add_mask_img).convert("RGB") if add_mask_img is not None else None
        add_mask_image = fit_to_resolution(add_mask_image, args.resolution)

        # Create mask tensors only if masks exist
        if mask_img is not None:
            mask_tensor = 2 * torch.tensor(np.array(mask_img)).float() / 255 - 1
            mask_tensor = rearrange(mask_tensor, "h w c -> 1 c h w").to(model.device)
        else:
            mask_tensor = None

        if add_mask_image is not None:
            add_mask_tensor = 2 * torch.tensor(np.array(add_mask_image)).float() / 255 - 1
            add_mask_tensor = rearrange(add_mask_tensor, "h w c -> 1 c h w").to(model.device)
        else:
            add_mask_tensor = None

        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([edict_instruction])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_tensor = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
            # cond["c_concat"] = [model.encode_first_stage(input_image).mode()]
            encoded_input = model.encode_first_stage(input_tensor).mode()

            if mask_tensor is not None or add_mask_tensor is not None:
                # Handle cases where at least one mask is provided
                if mask_tensor is not None and add_mask_tensor is not None:
                    # Both masks are provided
                    encoded_mask = model.encode_first_stage(mask_tensor).mode()
                    encoded_add_mask = model.encode_first_stage(add_mask_tensor).mode()
                    concat_cond = torch.cat([encoded_input, encoded_mask, encoded_add_mask], dim=1)
                elif mask_tensor is not None:
                    # Only mask_tensor (remove mask) is provided, use zeros for add_mask
                    encoded_mask = model.encode_first_stage(mask_tensor).mode()
                    zero_mask = torch.zeros_like(encoded_mask)
                    concat_cond = torch.cat([encoded_input, encoded_mask, zero_mask], dim=1)
                else:
                    # Only add_mask_tensor is provided, use zeros for remove mask
                    encoded_add_mask = model.encode_first_stage(add_mask_tensor).mode()
                    zero_mask = torch.zeros_like(encoded_add_mask)
                    concat_cond = torch.cat([encoded_input, zero_mask, encoded_add_mask], dim=1)
                
                cond["c_concat"] = [concat_cond]
            else:
                # Neither mask is provided but check if model expects the additional channels
                if hasattr(config.model.params, 'unet_config') and config.model.params.unet_config.params.in_channels == 16:
                    # Create two zero masks with same spatial dimensions as the input
                    zero_mask = torch.zeros_like(encoded_input)
                    concat_cond = torch.cat([encoded_input, zero_mask, zero_mask], dim=1)
                    cond["c_concat"] = [concat_cond]
                else:
                    # Original model without mask channels
                    cond["c_concat"] = [encoded_input]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(args.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0][:, :4]) * sigmas[0]  # Only take first 4 channels for latent
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        edited_image.save(output_path)
 
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
