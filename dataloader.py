# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
from typing import List, Tuple, Optional, Dict
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from class_sampler import build_tempered_weighted_sampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_train_tfms(img_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_tfms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _read_classes_txt(dataset_root: str) -> List[str]:
    p = os.path.join(dataset_root, "classes.txt")
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


class IP102FolderDataset(Dataset):
    """
    dataset_root/
      classes.txt
      classification/
        train/0..101/*.jpg
        val/0..101/*.jpg
        test/0..101/*.jpg
      superclasses/
        train/Alfalfa...Paddy/*.jpg
        val/Alfalfa...Paddy/*.jpg
        test/Alfalfa...Paddy/*.jpg

    split_dir = dataset_root/<images_subdir>/<split>/

    - If class folders are numeric: label = int(folder_name)
    - If class folders are strings: label = 0..K-1 via sorted folder names
    - classes.txt (if present) is used ONLY for numeric folders, or when its length matches K exactly.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str,
        img_size: int = 224,
        augment: Optional[bool] = None,
        images_subdir: str = "classification",
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.dataset_root = os.path.abspath(dataset_root)
        self.images_root = os.path.join(self.dataset_root, images_subdir)
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        split_dir = os.path.join(self.images_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        if augment is None:
            augment = (self.split == "train")

        self.transform = (
            build_train_tfms(img_size)
            if (self.split == "train" and augment)
            else build_eval_tfms(img_size)
        )

        self.class_names: List[str] = _read_classes_txt(self.dataset_root)
        self.idx_to_classname: Dict[int, str] = {}

        class_folders: List[str] = []
        for name in os.listdir(split_dir):
            p = os.path.join(split_dir, name)
            if os.path.isdir(p):
                class_folders.append(name)

        if not class_folders:
            raise RuntimeError(f"No class folders found in: {split_dir}")

        all_numeric = all(n.isdigit() for n in class_folders)
        any_numeric = any(n.isdigit() for n in class_folders)
        if any_numeric and not all_numeric:
            raise RuntimeError(
                f"Mixed numeric and non-numeric class folders in: {split_dir}. "
                f"Found: {sorted(class_folders)[:20]} ..."
            )

        if all_numeric:
            class_folders_sorted = sorted(class_folders, key=lambda s: int(s))
            self.class_to_idx: Dict[str, int] = {c: int(c) for c in class_folders_sorted}

            if self.class_names:
                max_idx = max(self.class_to_idx.values())
                if max_idx >= len(self.class_names):
                    raise ValueError(
                        f"classes.txt has {len(self.class_names)} lines but found class folder index {max_idx}."
                    )
                self.num_classes = len(self.class_names)
                self.idx_to_classname = {i: n for i, n in enumerate(self.class_names)}
            else:
                self.num_classes = max(self.class_to_idx.values()) + 1
                self.idx_to_classname = {}

        else:
            class_folders_sorted = sorted(class_folders)
            self.class_to_idx = {c: i for i, c in enumerate(class_folders_sorted)}
            self.num_classes = len(class_folders_sorted)

            self.idx_to_classname = {i: c for c, i in self.class_to_idx.items()}

            if self.class_names and len(self.class_names) == self.num_classes:
                self.idx_to_classname = {i: n for i, n in enumerate(self.class_names)}

        self.samples: List[Tuple[str, int]] = []
        for c in class_folders_sorted:
            y = self.class_to_idx[c]
            cdir = os.path.join(split_dir, c)
            for root_dir, _, files in os.walk(cdir):
                for fn in files:
                    if fn.lower().endswith(extensions):
                        self.samples.append((os.path.join(root_dir, fn), y))

        if not self.samples:
            raise RuntimeError(f"No images found under: {split_dir}")

        ys = [y for _, y in self.samples]
        y_min, y_max = min(ys), max(ys)
        if y_min < 0 or y_max >= self.num_classes:
            raise RuntimeError(
                f"Label out of range: min={y_min} max={y_max} but num_classes={self.num_classes}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = _open_rgb(path)
        img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)


def make_ip102_loader(
    dataset_root: str,
    split: str,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 8,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    sampler_alpha: float = 0.5,
    images_subdir="classification"
):
    ds = IP102FolderDataset(
        dataset_root=dataset_root,
        split=split,
        img_size=img_size,
        augment=(split.lower() == "train"),
        images_subdir=images_subdir
    )
    ys = [y for _, y in ds.samples]
    print(
        f"[DS] images_subdir={images_subdir} split={split} "
        f"num_classes={ds.num_classes} unique={len(set(ys))} "
        f"min={min(ys)} max={max(ys)}"
    )

    ys = [y for _, y in ds.samples]
    assert min(ys) >= 0 and max(ys) < ds.num_classes, (min(ys), max(ys), ds.num_classes)

    sampler = None
    if split.lower() == "train" and use_weighted_sampler:
        sampler, counts = build_tempered_weighted_sampler(ds, alpha=sampler_alpha)
        print(f"[Sampler] enabled | alpha={sampler_alpha} | classes={len(counts)} | "
              f"min_count={min(counts.values())} max_count={max(counts.values())}")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split.lower() == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split.lower() == "train"),
        persistent_workers=(num_workers > 0),
    )
    return ds, dl

