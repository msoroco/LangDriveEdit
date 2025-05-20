from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any

import PIL

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from datasets import load_dataset, load_from_disk, concatenate_datasets

class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)["edit"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            edit = prompt["edit"]
            input_prompt = prompt["input"]
            output_prompt = prompt["output"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)



class CarlaBoreasSimpleDataset(Dataset):
    def __init__(self,
        parquet_path : list,
        split="train",
        transform=None,
        seed = 42,
        # min_resize_res=256,
        # max_resize_res=256,
        # crop_res=256,
        # flip_prob=0.0,
        image_0_col="source_image",
        image_1_col="edited_image",
        prompt_col="edit_prompt",
        mask_col="mask_image",
        do_mask=True,
        resolution=[608, 368], # W, H
        center_crop=False,
        random_flip=False,
    ):
        datasets_list = []
        for path in parquet_path:
            print(f"Loading dataset from {path}...")
            dataset = load_dataset("parquet", data_files=path)
            if "train" in dataset:
                dataset_t = dataset["train"]
            elif "FreeForm" in dataset:
                # workaround for the freeform ultraedit dataset
                dataset_t = dataset["FreeForm"]
            else:
                dataset_t = dataset
            datasets_list.append(dataset_t)
            print(f"Loaded dataset from {path} with {len(dataset_t)} samples.")

        dataset = concatenate_datasets(datasets_list).shuffle(seed=seed)
        print(f"Concatenated dataset with {len(dataset)} samples.")
        self.ds = dataset

        # self.min_resize_res = min_resize_res
        # self.max_resize_res = max_resize_res
        # self.crop_res = crop_res
        # self.flip_prob = flip_prob
        self.image_0_col = image_0_col
        self.image_1_col = image_1_col
        self.prompt_col = prompt_col
        self.mask_col = mask_col
        self.do_mask = do_mask
        self.transform = transform
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # Load images from bytes or file paths
        image_0 = item[self.image_0_col]
        image_1 = item[self.image_1_col]
        prompt = item[self.prompt_col]
        W, H = self.resolution

        # If images are file paths, open them; if bytes, decode them
        if isinstance(image_0, str):
            image_0 = Image.open(image_0)
        elif isinstance(image_0, Image.Image):
            pass  # already a PIL Image
        else:
            image_0 = Image.open(BytesIO(image_0))
        if isinstance(image_1, str):
            image_1 = Image.open(image_1)
        elif isinstance(image_1, Image.Image):
            pass  # already a PIL Image
        else:
            image_1 = Image.open(BytesIO(image_1))


        image_0 = image_0.convert("RGB")
        image_1 = image_1.convert("RGB")
        image_0 = image_0.resize((W, H), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((W, H), Image.Resampling.LANCZOS)


        if self.do_mask:
            mask = item.get(self.mask_col, None)
            if mask is None:
                mask = PIL.Image.new("RGB", (W, H), (255, 255, 255))
            elif isinstance(mask, str):
                mask = Image.open(mask)
            elif isinstance(mask, Image.Image):
                pass  # already a PIL Image
            else:
                mask = Image.open(BytesIO(mask))
            mask = mask.convert("RGB") 
            mask = mask.resize((W, H), Image.Resampling.LANCZOS)
            mask = np.array(mask)
            mask = torch.tensor(mask)
            
            if mask.sum() == 0:  # if the mask is all 0, set it to 255
                mask = torch.ones_like(mask) * 255
            
        # reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        # image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        # image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        if self.do_mask:
            # mask = rearrange(2 * torch.tensor(mask).float() / 255 - 1, "h w c -> c h w")
            mask = torch.tensor(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)  # shape now [H, W, 1]
            mask = rearrange(2 * mask / 255 - 1, "h w c -> c h w")
        
        # crop = torchvision.transforms.RandomCrop(self.crop_res)
        crop = torchvision.transforms.RandomCrop((H, W))
        # flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        flip = torchvision.transforms.Lambda(lambda x: x)
        if self.do_mask:
            image_0, image_1, mask = flip(crop(torch.cat((image_0, image_1, mask)))).chunk(3)
        else:
            image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        # print(f"image_0 shape: {image_0.shape}")
        # print(f"image_1 shape: {image_1.shape}")
        # print(f"mask shape: {mask.shape}" if self.do_mask else "No mask provided")
        if self.do_mask:
            return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt), mask=mask)
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))
    


class CarlaBoreasFullDataset(Dataset):
    def __init__(self,
        parquet_path : list,
        split="train",
        transform=None,
        seed = 42,
        # min_resize_res=256,
        # max_resize_res=256,
        # crop_res=256,
        # flip_prob=0.0,
        image_0_col="source_image",
        image_1_col="edited_image",
        prompt_col="edit_prompt",
        remove_mask_col="source_mask",
        add_mask_col="edited_mask",
        do_mask=True,
        resolution=[608, 368], # W, H
        center_crop=False,
        random_flip=False,
    ):
        datasets_list = []
        for path in parquet_path:
            print(f"Loading dataset from {path}...")
            dataset = load_dataset("parquet", data_files=path)
            if "train" in dataset:
                dataset_t = dataset["train"]
            elif "FreeForm" in dataset:
                # workaround for the freeform ultraedit dataset
                dataset_t = dataset["FreeForm"]
            else:
                dataset_t = dataset
            datasets_list.append(dataset_t)
            print(f"Loaded dataset from {path} with {len(dataset_t)} samples.")

        dataset = concatenate_datasets(datasets_list).shuffle(seed=seed)
        print(f"Concatenated dataset with {len(dataset)} samples.")
        self.ds = dataset

        # self.min_resize_res = min_resize_res
        # self.max_resize_res = max_resize_res
        # self.crop_res = crop_res
        # self.flip_prob = flip_prob
        self.image_0_col = image_0_col
        self.image_1_col = image_1_col
        self.prompt_col = prompt_col
        self.add_mask_col = add_mask_col
        self.remove_mask_col = remove_mask_col
        self.do_mask = do_mask
        self.transform = transform
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # Load images from bytes or file paths
        image_0 = item[self.image_0_col]
        image_1 = item[self.image_1_col]

        if isinstance(item[self.prompt_col], list):
            random_num = torch.randint(0, len(item[self.prompt_col]), ()).item()
            prompt = item[self.prompt_col][random_num]  # sample one of the prompts
        else:
            prompt = item[self.prompt_col]  # use the string directly
        W, H = self.resolution

        # If images are file paths, open them; if bytes, decode them
        if isinstance(image_0, str):
            image_0 = Image.open(image_0)
        elif isinstance(image_0, Image.Image):
            pass  # already a PIL Image
        else:
            image_0 = Image.open(BytesIO(image_0))
        if isinstance(image_1, str):
            image_1 = Image.open(image_1)
        elif isinstance(image_1, Image.Image):
            pass  # already a PIL Image
        else:
            image_1 = Image.open(BytesIO(image_1))


        image_0 = image_0.convert("RGB")
        image_1 = image_1.convert("RGB")
        image_0 = image_0.resize((W, H), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((W, H), Image.Resampling.LANCZOS)


        if self.do_mask:
            remove_mask = item.get(self.remove_mask_col, None)
            add_mask = item.get(self.add_mask_col, None)
            def load_mask(mask):
                if mask is None:
                    mask = PIL.Image.new("RGB", (W, H), (255, 255, 255))
                elif isinstance(mask, str):
                    mask = Image.open(mask)
                elif isinstance(mask, Image.Image):
                    pass  # already a PIL Image
                else:
                    mask = Image.open(BytesIO(mask))
                mask = mask.convert("RGB") 
                mask = mask.resize((W, H), Image.Resampling.LANCZOS)
                mask = np.array(mask)
                mask = torch.tensor(mask)

                if mask.sum() == 0:  # if the mask is all 0, set it to 255
                    mask = torch.ones_like(mask) * 255
                return mask
            
            remove_mask = load_mask(remove_mask)
            add_mask = load_mask(add_mask)
            

        # reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        # image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        # image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        if self.do_mask:
            def transpose_mask(mask):
                mask = torch.tensor(mask).float()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(-1)  # shape now [H, W, 1]
                mask = rearrange(2 * mask / 255 - 1, "h w c -> c h w")
                return mask
            remove_mask = transpose_mask(remove_mask)
            add_mask = transpose_mask(add_mask)
        
        # crop = torchvision.transforms.RandomCrop(self.crop_res)
        crop = torchvision.transforms.RandomCrop((H, W))
        # flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        flip = torchvision.transforms.Lambda(lambda x: x)
        if self.do_mask:
            image_0, image_1, remove_mask, add_mask = flip(crop(torch.cat((image_0, image_1, remove_mask, add_mask)))).chunk(4)
        else:
            image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        # print(f"image_0 shape: {image_0.shape}")
        # print(f"image_1 shape: {image_1.shape}")
        # print(f"mask shape: {mask.shape}" if self.do_mask else "No mask provided")
        if self.do_mask:
            return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt), remove_mask=remove_mask, add_mask=add_mask)
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))
    

