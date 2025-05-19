from __future__ import annotations

import math
import os.path
import random
from argparse import ArgumentParser
from tqdm import tqdm

import einops
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import json
import os
from diffusers import StableDiffusion3InstructPix2PixPipeline

import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch


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



def load_model_from_config(ckpt, vae=None, text_encoder=None, text_encoder_2=None, 
                         text_encoder_3=None, transformer=None, revision=None, 
                         variant=None, torch_dtype=None):
    print(f"Loading model from {ckpt}")
    
    if all([vae, text_encoder, text_encoder_2, text_encoder_3, transformer]):
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            ckpt,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            transformer=transformer,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
    else:
        # Original loading logic
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(ckpt)
    return pipe



def main(args):
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # with open('your_input_path_dir/series_instructions.json') as fp:
    # with open(os.path.join(args.input_path + '-meta', 'series_instructions.json')) as fp:
    with open(os.path.join(args.input_path, '..', 'edit_sessions.json')) as fp:
        data_json = json.load(fp)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(1)

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    # Split data across GPUs
    items = list(data_json.items())
    chunk_size = len(items) // num_gpus
    data_chunks = {}
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(items)
        data_chunks[i] = dict(items[start_idx:end_idx])


     # Create processes for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=process_batch, 
                   args=(gpu_id, data_chunks[gpu_id], args))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

def process_batch(gpu_id, data_subset, args):
    # Set the device for this process
    torch.cuda.set_device(gpu_id)

    # Set environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pipe = load_model_from_config(args.ckpt)
    pipe = pipe.to(f"cuda:{gpu_id}")

    seed = random.randint(0, 100000) if args.seed is None else args.seed


    def fit_to_resolution(image, resolution):
        width, height = image.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

    def edit_image(input_path, output_path, edit_instruction, mask_img=None):
        torch.manual_seed(seed)
        input_image = Image.open(input_path).convert("RGB")
        input_image = fit_to_resolution(input_image, args.resolution)
        if mask_img is not None:
            mask_img = Image.open(mask_img).convert("RGB")
        else:
            mask_img = Image.new("RGB", input_image.size, (255, 255, 255))
        mask_img = fit_to_resolution(mask_img, args.resolution)
        # width, height = input_image.size
        # factor = args.resolution / max(width, height)
        # factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        # width = int((width * factor) // 64) * 64
        # height = int((height * factor) // 64) * 64
        # input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        edited_image = pipe(
            edit_instruction,
            image=input_image,
            mask_img=mask_img,
            negative_prompt="",
            num_inference_steps=args.steps,
            image_guidance_scale=args.cfg_image,
            guidance_scale=args.cfg_text,
        ).images[0]

        edited_image.save(output_path)

    if args.debug:
        data_subset = data_subset[:3]

    # iterative edit
    if args.skip_iter:
        print("Skip Iterative (Mutli-Turn) Editing......")
    else:
        print("Iterative (Mutli-Turn) Editing......")
        # for idx, datas in enumerate(data_json):
        for image_id, datas in tqdm(data_subset.items(), desc=f"GPU {gpu_id}"):
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
                    mask_image = data['mask']
                    mask_path = os.path.join(args.input_path, image_dir, mask_image)
                else:
                    mask_path = None
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
                    print('mask_path', mask_path)
                print('save_output_img_path', save_output_img_path)
                print('edit_instruction', edit_instruction)
                edit_image(image_path, save_output_img_path, edit_instruction, mask_path)

    print("Independent (Single-Turn) Editing......")
    # for idx, datas in enumerate(data_json):
    for image_id, datas in tqdm(data_subset.items(), desc=f"GPU {gpu_id}"):
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            image_dir = image_name.split('-')[0]
            image_path = os.path.join(args.input_path, image_dir, image_name)
            edit_instruction = data['instruction']
            if args.use_mask:
                mask_image = data['mask']
                mask_path = os.path.join(args.input_path, image_dir, mask_image)
            else:
                mask_path = None
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
                print('mask_path', mask_path)
            print('save_output_img_path', save_output_img_path)
            print('edit_instruction', edit_instruction)
            edit_image(image_path, save_output_img_path, edit_instruction, mask_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
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

    main(args)

