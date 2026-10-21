"""Temporal smoothing of live predictions (Experiment 6) and stability metrics.

Torch-free: works on numpy probability vectors, one per webcam frame.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np


class RawPredictor:
    """No smoothing: argmax of the current frame (the baseline for comparison)."""

    name = "raw"

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def update(self, probs: np.ndarray) -> tuple[int | None, float]:
        k = int(np.argmax(probs))
        conf = float(probs[k])
        return (k if conf >= self.threshold else None), conf

    def reset(self):
        pass


class MajorityVote:
    """Most frequent argmax label over the last `window` frames.

    The label is emitted only if it wins at least `min_agree` of the window and its mean
    confidence passes `threshold`; otherwise None ("no confident sign").
    """

    name = "majority"

    def __init__(self, window: int = 10, min_agree: float = 0.6, threshold: float = 0.5):
        self.window, self.min_agree, self.threshold = window, min_agree, threshold
        self.labels: deque[int] = deque(maxlen=window)
        self.confs: deque[float] = deque(maxlen=window)

    def update(self, probs: np.ndarray) -> tuple[int | None, float]:
        k = int(np.argmax(probs))
        self.labels.append(k)
        self.confs.append(float(probs[k]))
        label, count = Counter(self.labels).most_common(1)[0]
        conf = float(np.mean([c for l, c in zip(self.labels, self.confs) if l == label]))
        if count / self.window >= self.min_agree and conf >= self.threshold:
            return label, conf
        return None, conf

    def reset(self):
        self.labels.clear()
        self.confs.clear()


class EMASmoother:
    """Exponential moving average of the probability vectors, then argmax.

    alpha close to 1 = reacts fast (little smoothing); close to 0 = very smooth (more lag).
    """

    name = "ema"

    def __init__(self, alpha: float = 0.3, threshold: float = 0.5):
        self.alpha, self.threshold = alpha, threshold
        self.state: np.ndarray | None = None

    def update(self, probs: np.ndarray) -> tuple[int | None, float]:
        probs = np.asarray(probs, dtype=np.float64)
        self.state = probs if self.state is None else self.alpha * probs + (1 - self.alpha) * self.state
        k = int(np.argmax(self.state))
        conf = float(self.state[k])
        return (k if conf >= self.threshold else None), conf

    def reset(self):
        self.state = None


def build_smoother(name: str, **kw):
    if name == "raw":
        return RawPredictor(threshold=kw.get("threshold", 0.0))
    if name == "majority":
        return MajorityVote(window=kw.get("window", 10), min_agree=kw.get("min_agree", 0.6),
                            threshold=kw.get("threshold", 0.5))
    if name == "ema":
        return EMASmoother(alpha=kw.get("alpha", 0.3), threshold=kw.get("threshold", 0.5))
    raise ValueError(f"unknown smoother {name!r}")


# --------------------------------------------------------------------------- #
# Stability metrics
# --------------------------------------------------------------------------- #
def run_smoother(smoother, prob_seq: np.ndarray) -> list[int | None]:
    smoother.reset()
    return [smoother.update(p)[0] for p in prob_seq]


def stability_metrics(pred_seq: list[int | None], true_seq: list[int] | None = None,
                      fps: float = 30.0) -> dict:
    """Flicker and accuracy of a sequence of per-frame outputs.

    switches_per_min: how often the displayed label changes (lower = steadier)
    coverage:         fraction of frames where a label (not None) is shown
    frame_accuracy:   accuracy over frames with a known true label (None counts as wrong)
    latency_frames:   mean frames after each true-label change until the output first
                      shows the new label (the price paid for smoothing)
    """
    preds = list(pred_seq)
    n = len(preds)
    switches = sum(1 for a, b in zip(preds, preds[1:]) if a != b)
    out = {"frames": n,
           "switches": switches,
           "switches_per_min": switches / max(n / fps, 1e-9) * 60.0,
           "coverage": sum(p is not None for p in preds) / max(n, 1)}
    if true_seq is not None:
        true = list(true_seq)
        out["frame_accuracy"] = float(np.mean([p == t for p, t in zip(preds, true)]))
        lat = []
        change_points = [0] + [i for i in range(1, n) if true[i] != true[i - 1]]
        for i, c in enumerate(change_points):
            end = change_points[i + 1] if i + 1 < len(change_points) else n
            hit = next((j - c for j in range(c, end) if preds[j] == true[c]), None)
            if hit is not None:
                lat.append(hit)
        out["latency_frames"] = float(np.mean(lat)) if lat else float("nan")
        out["segments_recognised"] = f"{len(lat)}/{len(change_points)}"
    return out
