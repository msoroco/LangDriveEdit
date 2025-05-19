from __future__ import annotations

import math
import os.path
import random
import shutil
import sys
from argparse import ArgumentParser
from tqdm import tqdm

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
# sys.path.append("./stable_diffusion")

# from stable_diffusion.ldm.util import instantiate_from_config
# from ldm.util import instantiate_from_config
# from safetensors.torch import load_file
# from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


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
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_model_cpu_offload()
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

    # with open('your_input_path_dir/series_instructions.json') as fp:
    # with open(os.path.join(args.input_path + '-meta', 'series_instructions.json')) as fp:
    with open(os.path.join(args.input_path, '..', 'edit_sessions.json')) as fp:
        data_json = json.load(fp)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(1)
    # config = OmegaConf.load(args.config)
    pipe = load_model_from_config(args.ckpt)
    pipe = pipe.to("cuda")

    seed = random.randint(0, 100000) if args.seed is None else args.seed


    def fit_to_resolution(image, resolution):
        target_width, target_height = resolution
        width, height = image.size
        factor = min(target_width / width, target_height / height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

    def edit_image(input_path, output_path, edit_instruction, mask_img=None, add_mask_img=None):
        torch.manual_seed(seed)
        input_image = Image.open(input_path).convert("RGB")
        input_image = fit_to_resolution(input_image, args.resolution)

        mask_img = Image.open(mask_img).convert("L") if mask_img is not None else None
        add_mask_image = Image.open(add_mask_img).convert("L") if add_mask_img is not None else None
    
        width, height = input_image.size
        edited_image = pipe(
            edit_instruction,
            image=input_image,
            mask_img=mask_img,
            add_mask_image=add_mask_image,
            guidance_scale=args.cfg_text,
            num_inference_steps=args.steps,
            max_sequence_length=512,
            # generator = torch.Generator("cuda").manual_seed(seed)
        ).images[0]

        # edited_image = pipe(
        #     edit_instruction,
        #     image=input_image,
        #     mask_img=mask_img,
        #     negative_prompt="",
        #     num_inference_steps=args.steps,
        #     image_guidance_scale=args.cfg_image,
        #     guidance_scale=args.cfg_text,
        # ).images[0]

        edited_image.save(output_path)

    if args.debug:
        data_json = data_json[:3]

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
                    print('remove_mask_path', remove_mask_path)
                    print('add_mask_path', add_mask_path)
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
                print('remove_mask_path', remove_mask_path)
                print('add_mask_path', add_mask_path)
            print('save_output_img_path', save_output_img_path)
            print('edit_instruction', edit_instruction)
            edit_image(image_path, save_output_img_path, edit_instruction, remove_mask_path, add_mask_path)


if __name__ == "__main__":
    main()

