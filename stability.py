"""Experiment 6: does temporal smoothing reduce flicker, and what does it cost in delay?

Builds a frame stream with known ground truth, runs the trained model frame by frame,
then compares raw argmax with majority voting and EMA smoothing.

Stream sources
  (default) test set: each static test image is "held" in front of the camera for
            --hold frames with small per-frame jitter (camera shake, light, sensor noise);
            for dynamic runs, the stored frames of consecutive test clips are concatenated.
  --video FILE --labels FILE.csv   a real recording; labels CSV: start_frame,end_frame,label

python stability.py --run runs/<run> --num-samples 60 --hold 30
python stability.py --run runs/<run> --video my_signing.mp4 --labels my_labels.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from slr.data import read_rgb
from slr.engine import pick_device
from slr.live import LivePredictor
from slr.models import load_checkpoint
from slr.perturb import brightness, noise, translation
from slr.smoothing import build_smoother, run_smoother, stability_metrics
from slr.video import read_video_frames


def jitter(img, rng):
    """Small per-frame change, like a hand held still in front of a webcam."""
    img = translation(img, 1, rng) if rng.random() < 0.5 else img
    img = brightness(img, 1, rng) if rng.random() < 0.3 else img
    return noise(img, 1, rng)


def stream_from_test(ckpt, split_csv, root, n, hold, seed):
    df = pd.read_csv(split_csv)
    df = df[df.split == "test"]
    rng = np.random.default_rng(seed)
    pick = df.sample(min(n, len(df)), random_state=seed)
    idx = {c: i for i, c in enumerate(ckpt["classes"])}
    frames, truth = [], []
    for row in pick.itertuples():
        if ckpt["train_args"]["task"] == "static":
            img = read_rgb(Path(root) / row.path)
            for _ in range(hold):
                frames.append(jitter(img, rng))
                truth.append(idx[str(row.label)])
        else:
            d = Path(root) / row.path
            for f in sorted(d.glob("*.jpg")):
                frames.append(read_rgb(f))
                truth.append(idx[str(row.label)])
    return frames, truth


def stream_from_video(video, labels_csv, classes):
    frames = read_video_frames(video)
    lab = pd.read_csv(labels_csv)
    idx = {c: i for i, c in enumerate(classes)}
    truth = [-1] * len(frames)  # -1 = no sign / transition (excluded from accuracy)
    for r in lab.itertuples():
        for k in range(int(r.start_frame), min(int(r.end_frame) + 1, len(frames))):
            truth[k] = idx[str(r.label)]
    return frames, truth


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--num-samples", type=int, default=60)
    p.add_argument("--hold", type=int, default=30, help="frames per static sample")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--stride", type=int, default=0,
                   help="dynamic: CNN every N frames (0 = stored frames / seq_len)")
    p.add_argument("--windows", nargs="+", type=int, default=[5, 10, 15])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.3, 0.15])
    p.add_argument("--threshold", type=float, default=0.0,
                   help="confidence gate; >0 shows 'no sign' below it (counts as wrong)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    run = Path(a.run)
    device = pick_device(a.device)
    model, ckpt = load_checkpoint(run / "best.pt", device)
    ta = ckpt["train_args"]
    if a.video:
        if not a.labels:
            raise SystemExit("--video needs --labels")
        frames, truth = stream_from_video(a.video, a.labels, ckpt["classes"])
    else:
        frames, truth = stream_from_test(ckpt, ta["split_csv"], ta["data_root"], a.num_samples, a.hold, a.seed)
    stride = a.stride
    if ta["task"] == "dynamic" and stride == 0:
        stride = 1 if a.video else max(1, round(pd.read_csv(ta["split_csv"]).n_frames.median() / ta["seq_len"]))
    print(f"Stream: {len(frames)} frames, task={ta['task']}, stride={max(stride, 1)}")

    live = LivePredictor(model, ckpt, device, stride=max(stride, 1))
    n_cls = len(ckpt["classes"])
    probs = []
    for f in frames:
        pr = live.step(f)
        probs.append(pr if pr is not None else np.full(n_cls, 1.0 / n_cls))
    probs = np.stack(probs)
    np.save(run / "stability_probs.npy", probs)

    keep = np.array([t >= 0 for t in truth])
    configs = [("raw", {})] + [("majority", {"window": w}) for w in a.windows] + \
              [("ema", {"alpha": al}) for al in a.alphas]
    rows, traces = [], {}
    for name, kw in configs:
        sm = build_smoother(name, threshold=a.threshold, **kw)
        pred = run_smoother(sm, probs)
        label = name + "".join(f" {k}={v}" for k, v in kw.items())
        traces[label] = pred
        m = stability_metrics([p for p, k in zip(pred, keep) if k],
                              [t for t, k in zip(truth, keep) if k], fps=a.fps)
        rows.append({"smoother": label, **m})
    df = pd.DataFrame(rows)
    df.to_csv(run / "stability.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_trace(truth, traces, ckpt["classes"], run / "stability_trace.png", a.windows[len(a.windows) // 2])
    print(f"Saved {run / 'stability.csv'} and stability_trace.png")


def plot_trace(truth, traces, classes, path, window, max_frames=450):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["raw", f"majority window={window}"]
    show = [s for s in show if s in traces]
    n = min(max_frames, len(truth))
    fig, ax = plt.subplots(len(show) + 1, 1, figsize=(12, 1.6 * (len(show) + 1)), sharex=True)
    series = [("ground truth", truth)] + [(s, traces[s]) for s in show]
    for axis, (name, seq) in zip(ax, series):
        y = np.array([np.nan if (v is None or v < 0) else v for v in seq[:n]], dtype=float)
        axis.step(range(n), y, where="post", lw=1.2)
        axis.set_ylabel(name, rotation=0, ha="right", fontsize=8)
        axis.set_yticks([])
    ax[-1].set_xlabel("frame")
    fig.suptitle("Predicted class over time (flat = stable, spikes = flicker)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
