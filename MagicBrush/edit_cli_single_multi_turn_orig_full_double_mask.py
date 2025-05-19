from __future__ import annotations

import math
import os.path
import random
import shutil
import sys
from argparse import ArgumentParser
from tqdm import tqdm

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

    # with open('your_input_path_dir/series_instructions.json') as fp:
    # with open(os.path.join(args.input_path + '-meta', 'series_instructions.json')) as fp:
    with open(os.path.join(args.input_path, '..', 'edit_sessions.json')) as fp:
        data_json = json.load(fp)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(1)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed


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

        mask_tensor = 2 * torch.tensor(np.array(mask_img)).float() / 255 - 1
        mask_tensor = rearrange(mask_tensor, "h w c -> 1 c h w").to(model.device)

        add_mask_tensor = 2 * torch.tensor(np.array(add_mask_image)).float() / 255 - 1
        add_mask_tensor = rearrange(add_mask_tensor, "h w c -> 1 c h w").to(model.device)

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

    if args.debug:
        data_json = data_json[:23]

    # iterative edit
    if args.skip_iter:
        print("Skip Iterative (Mutli-Turn) Editing......")
    else:
        print("Iterative (Mutli-Turn) Editing......")
        # for idx, datas in enumerate(data_json):
        for image_id, datas in tqdm(data_json.items()):
            for turn_id, data in enumerate(datas):
                print("data", data)
                image_name = data['input']
                image_dir = image_name.split('-')[0]
                # 139306-input.png
                # 139306-output1.png
                # image_dir is 139306
                if turn_id == 0:  # first enter
                    image_path = os.path.join(args.input_path, image_dir, image_name)
                else:
                    image_path = save_output_img_path
                edit_instruction = data['instruction']
                if args.use_mask:
                    add_mask_image = data['add_mask']
                    remove_mask_image = data['remove_mask']
                    add_mask_path = os.path.join(args.input_path, image_dir, add_mask_image)
                    remove_mask_path = os.path.join(args.input_path, image_dir, remove_mask_image)
                else:
                    add_mask_path = None
                    remove_mask_path = None
                save_output_dir_path = os.path.join(args.output_path, image_dir)
                if not os.path.exists(save_output_dir_path):
                    os.makedirs(save_output_dir_path)
                if turn_id == 0:
                    save_output_img_path = os.path.join(save_output_dir_path, image_dir+'_1.png')
                else:
                    save_output_img_path = os.path.join(save_output_dir_path, image_dir+'_iter_' +str(turn_id + 1)+'.png')
                
                if os.path.exists(save_output_img_path):
                    print('already generated, skip')
                    continue
                print('image_name', image_name)
                print('image_path', image_path)
                if args.use_mask:
                    print('add_mask_path', add_mask_path)
                    print('remove_mask_path', remove_mask_path)
                print('save_output_img_path', save_output_img_path)
                print('edit_instruction', edit_instruction)
                edit_image(image_path, save_output_img_path, edit_instruction, remove_mask_path, add_mask_path)

    print("Independent (Single-Turn) Editing......")
    # for idx, datas in enumerate(data_json):
    for image_id, datas in tqdm(data_json.items()):
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            image_dir = image_name.split('-')[0]
            image_path = os.path.join(args.input_path, image_dir, image_name)
            edit_instruction = data['instruction']
            if args.use_mask:
                add_mask_image = data['add_mask']
                remove_mask_image = data['remove_mask']
                add_mask_path = os.path.join(args.input_path, image_dir, add_mask_image)
                remove_mask_path = os.path.join(args.input_path, image_dir, remove_mask_image)
            else:
                add_mask_path = None
                remove_mask_path = None
            save_outut_dir_path = os.path.join(args.output_path, image_dir)
            if not os.path.exists(save_outut_dir_path):
                os.makedirs(save_outut_dir_path)
            if turn_id == 0:
                save_output_img_path = os.path.join(save_outut_dir_path, image_dir+'_1.png')
                # if os.path.exists(save_output_img_path):  # already generated in iterative editing
                #     print('already generated in iterative editing')
                #     continue
            else:
                save_output_img_path = os.path.join(save_outut_dir_path, image_dir+'_inde_' +str(turn_id + 1)+'.png')
            if os.path.exists(save_output_img_path):  # already generated in iterative (multi-turn) editing.
                print('already generated, skip')
                continue
            print('image_name', image_name)
            print('image_path', image_path)
            if args.use_mask:
                print('add_mask_path', add_mask_path)
                print('remove_mask_path', remove_mask_path)
            print('save_output_img_path', save_output_img_path)
            print('edit_instruction', edit_instruction)
            edit_image(image_path, save_output_img_path, edit_instruction, remove_mask_path, add_mask_path)


if __name__ == "__main__":
    main()
