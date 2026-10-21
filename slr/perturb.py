"""Controlled image perturbations for the robustness experiment (Experiment 3).

All functions take an RGB uint8 image (H, W, 3) and return one of the same shape.
Parameters depend only on (severity, rng), so passing an rng seeded per *clip* applies
the identical perturbation to every frame of a video, as a real camera change would.

Severity levels 1-3 are mild / moderate / strong.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

Perturbation = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def _affine(img: np.ndarray, angle: float = 0.0, scale: float = 1.0,
            tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx * w
    m[1, 2] += ty * h
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def brightness(img, severity, rng):
    delta = [0, 40, 70, 100][severity] * rng.choice([-1, 1])
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def contrast(img, severity, rng):
    factor = [1.0, 0.7, 0.5, 0.3][severity]
    mean = img.mean(axis=(0, 1), keepdims=True)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)


def rotation(img, severity, rng):
    return _affine(img, angle=[0, 10, 20, 30][severity] * rng.choice([-1, 1]))


def scale(img, severity, rng):
    s = [1.0, 0.85, 0.7, 0.55][severity]          # hand appears smaller / further away
    return _affine(img, scale=s if rng.random() < 0.5 else 2 - s)


def translation(img, severity, rng):
    d = [0, 0.08, 0.15, 0.22][severity]
    ang = rng.uniform(0, 2 * np.pi)
    return _affine(img, tx=d * np.cos(ang), ty=d * np.sin(ang))


def blur(img, severity, rng):
    k = [1, 5, 9, 15][severity]
    return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img


def noise(img, severity, rng):
    sigma = [0, 10, 20, 35][severity]
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def occlusion(img, severity, rng):
    h, w = img.shape[:2]
    frac = [0, 0.15, 0.25, 0.35][severity]
    ph, pw = int(h * frac), int(w * frac)
    if ph == 0:
        return img
    y, x = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
    out = img.copy()
    out[y:y + ph, x:x + pw] = 0
    return out


def low_resolution(img, severity, rng):
    f = [1, 2, 4, 6][severity]                      # simulates a poor webcam
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


PERTURBATIONS: dict[str, Perturbation] = {
    "brightness": brightness, "contrast": contrast, "rotation": rotation, "scale": scale,
    "translation": translation, "blur": blur, "noise": noise, "occlusion": occlusion,
    "low_resolution": low_resolution,
}


class BackgroundReplacer:
    """Background-variation test. Needs a foreground mask per image (white = hand/signer).

    Masks can come from a segmentation model (e.g. MediaPipe selfie segmentation) or a
    green-screen recording; masks/<same relative path>.png. Backgrounds are any images.
    Severity controls how cluttered (high-frequency) the chosen background is.
    """

    def __init__(self, mask_root: str | Path, background_dir: str | Path):
        self.mask_root = Path(mask_root)
        bgs = [p for p in Path(background_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not bgs:
            raise FileNotFoundError(f"No background images in {background_dir}")
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in bgs]
        # order backgrounds by clutter (mean gradient magnitude)
        clutter = [float(np.abs(cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY), cv2.CV_32F)).mean())
                   for i in imgs]
        self.backgrounds = [imgs[i] for i in np.argsort(clutter)]

    def __call__(self, img: np.ndarray, severity: int, rng: np.random.Generator,
                 rel_path: str) -> np.ndarray:
        mask_path = (self.mask_root / rel_path).with_suffix(".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        h, w = img.shape[:2]
        n = len(self.backgrounds)
        lo, hi = int((severity - 1) / 3 * n), max(int(severity / 3 * n), int((severity - 1) / 3 * n) + 1)
        bg = self.backgrounds[int(rng.integers(lo, min(hi, n)))]
        bg = cv2.resize(bg, (w, h))
        alpha = cv2.GaussianBlur(cv2.resize(mask, (w, h)).astype(np.float32) / 255.0, (5, 5), 0)
        alpha = alpha[..., None]
        return (img * alpha + bg * (1 - alpha)).astype(np.uint8)


class DatasetPerturbation:
    """Picklable adapter to the dataset hook perturb(img, rng, rel_path).

    (A class rather than a lambda so it works with DataLoader workers on Windows/macOS,
    which pickle the dataset.)
    """

    def __init__(self, fn, severity: int, needs_path: bool = False):
        self.fn, self.severity, self.needs_path = fn, severity, needs_path

    def __call__(self, img: np.ndarray, rng: np.random.Generator, rel_path: str) -> np.ndarray:
        if self.needs_path:
            return self.fn(img, self.severity, rng, rel_path)
        return self.fn(img, self.severity, rng)


def apply(name: str, img: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Convenience wrapper: deterministic perturbation of one image."""
    if severity == 0:
        return img
    return PERTURBATIONS[name](img, severity, np.random.default_rng(seed))
