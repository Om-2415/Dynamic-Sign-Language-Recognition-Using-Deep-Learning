"""Video helpers: frame extraction and temporal sampling (torch-free)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_video_frames(path: str | Path) -> list[np.ndarray]:
    """Read all frames of a video as RGB uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def resize_short_side(img: np.ndarray, short: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = short / min(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def extract_frames(video_path: str | Path, out_dir: str | Path, num_frames: int = 32,
                   short_side: int = 256, start: int | None = None, end: int | None = None) -> int:
    """Save `num_frames` uniformly spaced frames as JPEGs (00000.jpg ...).

    Storing a fixed number of frames per clip keeps disk usage bounded; training then
    samples `seq_len` (<= num_frames) of them with temporal jitter.
    start/end: optional frame range (WLASL gives frame_start/frame_end per instance).
    Returns the number of frames written (0 if the video could not be read).
    """
    frames = read_video_frames(video_path)
    if start is not None or end is not None:
        frames = frames[(start or 0):(end if end and end > 0 else None)]
    if not frames:
        return 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = uniform_indices(len(frames), num_frames)
    for i, k in enumerate(idx):
        fr = resize_short_side(frames[k], short_side)
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(idx)


def uniform_indices(n_frames: int, seq_len: int) -> np.ndarray:
    """Evenly spaced indices covering the whole clip (repeats frames if the clip is short)."""
    if n_frames <= 0:
        raise ValueError("empty clip")
    return np.linspace(0, n_frames - 1, seq_len).round().astype(int)


def sample_indices(n_frames: int, seq_len: int, train: bool, rng: np.random.Generator | None = None,
                   min_span: float = 0.8) -> np.ndarray:
    """Temporal sampling. Eval: uniform over the clip. Train: random sub-span covering
    >= min_span of the clip, plus per-segment jitter -> robustness to signing speed.
    Indices are always sorted, so temporal order is preserved.
    """
    if not train or n_frames <= 1:
        return uniform_indices(n_frames, seq_len)
    rng = rng or np.random.default_rng()
    span = int(round(n_frames * rng.uniform(min_span, 1.0)))
    span = max(span, 1)
    start = int(rng.integers(0, n_frames - span + 1))
    edges = np.linspace(start, start + span, seq_len + 1)
    idx = [int(rng.integers(int(a), max(int(a) + 1, int(b)))) for a, b in zip(edges[:-1], edges[1:])]
    return np.clip(np.sort(idx), 0, n_frames - 1)
