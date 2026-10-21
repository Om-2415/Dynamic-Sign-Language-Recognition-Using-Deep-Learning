"""PyTorch datasets and transforms for static images and frame sequences."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from .video import sample_indices

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# perturb(img_rgb_uint8, rng, rel_path) -> img_rgb_uint8   (used by robustness / stability)
PerturbFn = Callable[[np.ndarray, np.random.Generator, str], np.ndarray]


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_transforms(img_size: int, train: bool, aug: str = "standard", hflip: bool = False):
    """Augmentation is used ONLY for training. Evaluation and live inference use the
    deterministic branch (resize + normalise), matching the corrected architecture.

    hflip is off by default: mirroring changes the dominant hand and can change the
    meaning of direction-dependent signs. Enable it only as a deliberate experiment.
    """
    norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train or aug == "none":
        return v2.Compose([v2.Resize((img_size, img_size), antialias=True), *norm])
    ops = [
        v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True),
        v2.RandomApply([v2.RandomAffine(degrees=15, translate=(0.08, 0.08))], p=0.7),
        v2.RandomPerspective(distortion_scale=0.15, p=0.3),
        v2.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03),
    ]
    if hflip:
        ops.append(v2.RandomHorizontalFlip())
    ops += [*norm, v2.RandomErasing(p=0.25, scale=(0.02, 0.12))]
    return v2.Compose(ops)


def class_list(df: pd.DataFrame) -> list[str]:
    return sorted(df["label"].astype(str).unique())


class StaticSignDataset(Dataset):
    """One image -> one label. `root` + df.path gives the image file."""

    def __init__(self, df: pd.DataFrame, root: str | Path, classes: list[str], transform,
                 perturb: PerturbFn | None = None, perturb_seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cls_idx = {c: i for i, c in enumerate(classes)}
        self.transform, self.perturb, self.perturb_seed = transform, perturb, perturb_seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = read_rgb(self.root / row.path)
        if self.perturb is not None:
            img = self.perturb(img, np.random.default_rng(self.perturb_seed + i), row.path)
        x = self.transform(tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)))
        return x.as_subclass(torch.Tensor), self.cls_idx[str(row.label)]


class VideoSignDataset(Dataset):
    """A folder of ordered frames -> one label. Returns a (T, C, H, W) tensor.

    All frames of a clip receive the SAME random augmentation (tv_tensors.Video), so
    augmentation never creates fake motion between frames.
    """

    def __init__(self, df: pd.DataFrame, root: str | Path, classes: list[str], seq_len: int,
                 transform, train: bool, perturb: PerturbFn | None = None, perturb_seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cls_idx = {c: i for i, c in enumerate(classes)}
        self.seq_len, self.transform, self.train = seq_len, transform, train
        self.perturb, self.perturb_seed = perturb, perturb_seed
        self._frames: dict[int, list[Path]] = {}

    def __len__(self):
        return len(self.df)

    def frame_paths(self, i) -> list[Path]:
        if i not in self._frames:
            d = self.root / self.df.iloc[i].path
            fr = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not fr:
                raise FileNotFoundError(f"No frames in {d}")
            self._frames[i] = fr
        return self._frames[i]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        paths = self.frame_paths(i)
        idx = sample_indices(len(paths), self.seq_len, self.train)
        frames = [read_rgb(paths[k]) for k in idx]
        if self.perturb is not None:  # same perturbation for every frame of the clip
            frames = [self.perturb(f, np.random.default_rng(self.perturb_seed + i), row.path) for f in frames]
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # T,C,H,W
        x = self.transform(tv_tensors.Video(clip))
        return x.as_subclass(torch.Tensor), self.cls_idx[str(row.label)]


def make_dataset(task: str, df: pd.DataFrame, root, classes, img_size: int, train: bool,
                 seq_len: int = 16, aug: str = "standard", hflip: bool = False,
                 perturb: PerturbFn | None = None):
    tf = build_transforms(img_size, train=train, aug=aug, hflip=hflip)
    if task == "static":
        return StaticSignDataset(df, root, classes, tf, perturb=perturb)
    if task == "dynamic":
        return VideoSignDataset(df, root, classes, seq_len, tf, train=train, perturb=perturb)
    raise ValueError(f"unknown task {task!r}")
