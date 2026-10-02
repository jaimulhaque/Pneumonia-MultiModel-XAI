"""Shared data pipeline: CLAHE -> aspect-ratio resize+pad (384) -> 3ch -> ImageNet norm."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def clahe_resize_pad(gray, size=IMG_SIZE, clip=2.0, tile=8):
    """gray uint8 HxW -> CLAHE -> keep aspect ratio -> zero-pad to size x size."""
    g = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
    h, w = g.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    g = cv2.resize(g, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = g
    return canvas


def cached_path(cache_dir, rel_path):
    return Path(cache_dir) / Path(rel_path).with_suffix(".png")


def build_cache(df, raw_dir, cache_dir, size=IMG_SIZE, overwrite=False):
    """Run CLAHE + resize + pad once and store lossless PNGs (fast training)."""
    n_new = 0
    for rel in df["image_path"]:
        out = cached_path(cache_dir, rel)
        if out.exists() and not overwrite:
            continue
        gray = cv2.imread(str(Path(raw_dir) / rel), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read {rel}")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), clahe_resize_pad(gray, size))
        n_new += 1
    return n_new


def build_train_aug():
    """Identical for every model. No random erasing."""
    return v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
    ])


class CXRDataset(Dataset):
    def __init__(self, csv_path, cache_dir, train=False):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.paths = [cached_path(cache_dir, p) for p in self.df["image_path"]]
        self.labels = [CLASS_TO_IDX[l] for l in self.df["label"]]
        self.aug = build_train_aug() if train else None
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        if self.aug is not None:
            img = self.aug(img)
        img = img.convert("RGB")                 # grayscale -> 3 channels
        return self.to_tensor(img), self.labels[i]


def class_weights(labels, n_classes=len(CLASSES)):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def make_mix_collate(num_classes=len(CLASSES), alpha=1.0):
    """Optional MixUp/CutMix (random pick per batch). Returns soft labels."""
    mix = v2.RandomChoice([v2.CutMix(alpha=alpha, num_classes=num_classes),
                           v2.MixUp(alpha=alpha, num_classes=num_classes)])
    def collate(batch):
        x, y = torch.utils.data.default_collate(batch)
        return mix(x, y)
    return collate


def _worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


def make_loaders(fold, splits_dir, cache_dir, batch_size=16, num_workers=4,
                 seed=42, use_mix=False):
    splits_dir = Path(splits_dir)
    ds = {r: CXRDataset(splits_dir / f"fold{fold}_{r}.csv", cache_dir, train=(r == "train"))
          for r in ["train", "val", "test"]}
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True, generator=g,
                              worker_init_fn=_worker_init,
                              collate_fn=make_mix_collate() if use_mix else None,
                              persistent_workers=num_workers > 0)
    eval_kw = dict(batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
                   pin_memory=True, persistent_workers=num_workers > 0)
    return (train_loader, DataLoader(ds["val"], **eval_kw), DataLoader(ds["test"], **eval_kw),
            class_weights(ds["train"].labels), ds)
