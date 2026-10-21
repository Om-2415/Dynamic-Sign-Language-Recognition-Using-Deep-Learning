"""Generate a tiny synthetic dataset to smoke-test the whole pipeline in minutes.

NOT sign language. It only exercises the code paths:
  static:  8 "hand shape" classes drawn by 6 "signers" (each signer has own colour,
           stroke width, size and background) -> tests random vs signer-independent splits.
  dynamic: 6 motion classes (left->right, right->left, up, down, clockwise, anticlockwise).
           Pairs like left->right / right->left contain the same frames in opposite order,
           so a frame-averaging model ('--rnn none') cannot separate them but an LSTM can.

python scripts/make_synthetic_data.py --out data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SHAPES = ["circle", "square", "triangle", "star", "cross", "ring", "two_dots", "bar"]
MOTIONS = ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "clockwise", "anticlockwise"]


def signer_style(sid: int):
    r = np.random.default_rng(1000 + sid)
    return {"color": tuple(int(c) for c in r.integers(90, 255, 3)), "thick": int(r.integers(2, 6)),
            "scale": float(r.uniform(0.75, 1.15)),
            "bg": tuple(int(c) for c in r.integers(0, 90, 3))}


def background(size, base, rng):
    img = np.full((size, size, 3), base, np.uint8).astype(np.int16)
    img += rng.integers(-25, 25, (size, size, 3))
    for _ in range(3):  # clutter
        p = tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.circle(img, p, int(rng.integers(3, 12)), tuple(int(c) for c in rng.integers(0, 120, 3)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_shape(img, shape, c, r, color, thick):
    x, y = c
    if shape == "circle":
        cv2.circle(img, c, r, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, -1)
    elif shape == "triangle":
        pts = np.array([[x, y - r], [x - r, y + r], [x + r, y + r]], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == "star":
        ang = np.linspace(0, 2 * np.pi, 11)[:-1] - np.pi / 2
        rad = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
        pts = np.stack([x + rad * np.cos(ang), y + rad * np.sin(ang)], 1).astype(np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == "cross":
        cv2.line(img, (x - r, y), (x + r, y), color, thick * 2)
        cv2.line(img, (x, y - r), (x, y + r), color, thick * 2)
    elif shape == "ring":
        cv2.circle(img, c, r, color, thick * 2)
    elif shape == "two_dots":
        cv2.circle(img, (x - r // 2, y), r // 3, color, -1)
        cv2.circle(img, (x + r // 2, y), r // 3, color, -1)
    elif shape == "bar":
        cv2.rectangle(img, (x - r // 4, y - r), (x + r // 4, y + r), color, -1)


def make_static(out: Path, signers: int, per_class: int, size: int):
    for sid in range(signers):
        st = signer_style(sid)
        rng = np.random.default_rng(sid)
        for shape in SHAPES:
            d = out / f"signer{sid}" / shape
            d.mkdir(parents=True, exist_ok=True)
            for k in range(per_class):
                img = background(size, st["bg"], rng)
                r = int(size * 0.28 * st["scale"] * rng.uniform(0.85, 1.15))
                c = tuple(int(v) for v in (size / 2 + rng.normal(0, size * 0.06, 2)))
                draw_shape(img, shape, c, r, st["color"], st["thick"])
                if rng.random() < 0.5:  # small rotation
                    m = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-15, 15), 1)
                    img = cv2.warpAffine(img, m, (size, size), borderMode=cv2.BORDER_REFLECT_101)
                cv2.imwrite(str(d / f"{shape}_{sid}_{k:03d}.png"), img)


def trajectory(motion, n, size, rng):
    t = np.linspace(0, 1, n)
    lo, hi = size * 0.2, size * 0.8
    jitter = rng.normal(0, size * 0.01, (n, 2))
    mid = size / 2 + rng.normal(0, size * 0.04)
    if motion == "left_to_right":
        xy = np.stack([lo + (hi - lo) * t, np.full(n, mid)], 1)
    elif motion == "right_to_left":
        xy = np.stack([hi - (hi - lo) * t, np.full(n, mid)], 1)
    elif motion == "top_to_bottom":
        xy = np.stack([np.full(n, mid), lo + (hi - lo) * t], 1)
    elif motion == "bottom_to_top":
        xy = np.stack([np.full(n, mid), hi - (hi - lo) * t], 1)
    else:
        sgn = 1 if motion == "clockwise" else -1
        a = sgn * 2 * np.pi * t
        xy = np.stack([size / 2 + size * 0.28 * np.cos(a), size / 2 + size * 0.28 * np.sin(a)], 1)
    return (xy + jitter).astype(int)


def make_dynamic(out: Path, signers: int, per_class: int, size: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for sid in range(signers):
        st = signer_style(sid)
        rng = np.random.default_rng(100 + sid)
        for motion in MOTIONS:
            d = out / f"signer{sid}" / motion
            d.mkdir(parents=True, exist_ok=True)
            for k in range(per_class):
                n = int(rng.integers(24, 41))  # different signing speeds
                bg = background(size, st["bg"], rng)
                w = cv2.VideoWriter(str(d / f"{motion}_{sid}_{k:02d}.mp4"), fourcc, 25, (size, size))
                r = int(size * 0.1 * st["scale"])
                for x, y in trajectory(motion, n, size, rng):
                    fr = bg.copy()
                    cv2.circle(fr, (int(x), int(y)), r, st["color"], -1)
                    w.write(fr)
                w.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--signers", type=int, default=6)
    p.add_argument("--static-per-class", type=int, default=25)
    p.add_argument("--videos-per-class", type=int, default=6)
    p.add_argument("--size", type=int, default=96)
    a = p.parse_args()
    out = Path(a.out)
    make_static(out / "static", a.signers, a.static_per_class, a.size)
    make_dynamic(out / "videos", a.signers, a.videos_per_class, a.size)
    print(f"Synthetic data written to {out}/static and {out}/videos")


if __name__ == "__main__":
    main()
