"""Experiment 7: real-time OpenCV prototype.

Webcam (or video file) -> preprocessing -> model -> temporal smoothing -> on-screen label.

python webcam.py --run runs/<static run>                     # hand inside the green box
python webcam.py --run runs/<dynamic run> --stride 4          # full frame, sliding window
python webcam.py --run runs/<run> --source demo.mp4 --save-video out.mp4 --no-display

Keys: q = quit, r = reset buffer/smoother, space = pause, m = toggle mirror
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from slr.engine import pick_device
from slr.live import LivePredictor
from slr.models import load_checkpoint
from slr.smoothing import build_smoother


def roi_box(w, h, frac, cx, cy):
    s = int(min(w, h) * frac)
    x0 = int(np.clip(cx * w - s / 2, 0, w - s))
    y0 = int(np.clip(cy * h - s / 2, 0, h - s))
    return x0, y0, s


def put(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--source", default="0", help="camera index or video file")
    p.add_argument("--smoother", choices=["raw", "majority", "ema"], default="majority")
    p.add_argument("--window", type=int, default=10, help="majority-vote window (frames)")
    p.add_argument("--alpha", type=float, default=0.3, help="EMA factor")
    p.add_argument("--threshold", type=float, default=0.6, help="below this, show '...' instead of a label")
    p.add_argument("--stride", type=int, default=4, help="dynamic: run CNN every N frames")
    p.add_argument("--roi", type=float, default=0.55, help="static: ROI box size as fraction of frame")
    p.add_argument("--roi-center", type=float, nargs=2, default=[0.7, 0.45], help="ROI centre (x y fractions)")
    p.add_argument("--mirror", action="store_true", help="flip input horizontally (selfie view)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--save-video", default=None)
    p.add_argument("--log", default=None, help="CSV of per-frame raw/smoothed predictions and latency")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    device = pick_device(a.device)
    model, ckpt = load_checkpoint(Path(a.run) / "best.pt", device)
    classes = ckpt["classes"]
    live = LivePredictor(model, ckpt, device, stride=a.stride)
    smoother = build_smoother(a.smoother, window=a.window, alpha=a.alpha, threshold=a.threshold)
    static = live.task == "static"

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {a.source!r}")

    writer, log_f, log_w = None, None, None
    if a.log:
        log_f = open(a.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["frame", "raw", "raw_conf", "smoothed", "smoothed_conf", "infer_ms"])

    mirror, paused, frame_i = a.mirror, False, 0
    fps_hist: deque[float] = deque(maxlen=30)
    ms_hist: deque[float] = deque(maxlen=30)
    print(f"Running {ckpt['run_name']} ({live.task}, {len(classes)} classes) on {device}. q to quit.")
    try:
        while True:
            if not paused:
                t_frame = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if static:
                    x0, y0, s = roi_box(w, h, a.roi, *a.roi_center)
                    inp = rgb[y0:y0 + s, x0:x0 + s]
                else:
                    inp = rgb
                t0 = time.perf_counter()
                probs = live.step(inp)
                infer_ms = (time.perf_counter() - t0) * 1000
                ms_hist.append(infer_ms)

                raw_lab, raw_conf, sm_lab, sm_conf = None, 0.0, None, 0.0
                if probs is not None:
                    raw_lab = int(np.argmax(probs))
                    raw_conf = float(probs[raw_lab])
                    sm_lab, sm_conf = smoother.update(probs)

                vis = frame.copy()
                if static:
                    cv2.rectangle(vis, (x0, y0), (x0 + s, y0 + s), (0, 220, 0), 2)
                    put(vis, "place hand in box", (x0, y0 - 10), 0.6, (0, 220, 0))
                if probs is None:
                    put(vis, f"buffering {len(live.feats)}/{live.seq_len}", (20, 40))
                else:
                    shown = classes[sm_lab] if sm_lab is not None else "..."
                    put(vis, f"{shown}", (20, 70), 2.0, (0, 255, 255), 4)
                    put(vis, f"conf {sm_conf:.2f}  ({a.smoother})", (20, 110), 0.7)
                    put(vis, f"raw: {classes[raw_lab]} {raw_conf:.2f}", (20, 140), 0.6, (200, 200, 200))
                    top = np.argsort(-probs)[:3]
                    for k, c in enumerate(top):
                        y = h - 90 + k * 25
                        cv2.rectangle(vis, (20, y - 15), (20 + int(200 * probs[c]), y + 3), (255, 160, 0), -1)
                        put(vis, f"{classes[c]} {probs[c]:.2f}", (230, y), 0.55)
                fps_hist.append(1.0 / max(time.perf_counter() - t_frame, 1e-6))
                put(vis, f"{np.mean(fps_hist):.1f} FPS  model {np.mean(ms_hist):.1f} ms", (w - 330, 30), 0.6)

                if a.save_video:
                    if writer is None:
                        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
                        writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
                    writer.write(vis)
                if log_w:
                    log_w.writerow([frame_i, classes[raw_lab] if raw_lab is not None else "",
                                    f"{raw_conf:.4f}", classes[sm_lab] if sm_lab is not None else "",
                                    f"{sm_conf:.4f}", f"{infer_ms:.2f}"])
                frame_i += 1
            if not a.no_display:
                cv2.imshow("Sign recognition", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    live.reset()
                    smoother.reset()
                if key == ord(" "):
                    paused = not paused
                if key == ord("m"):
                    mirror = not mirror
    finally:
        cap.release()
        if writer:
            writer.release()
        if log_f:
            log_f.close()
        cv2.destroyAllWindows() if not a.no_display else None
    if ms_hist:
        print(f"Processed {frame_i} frames; mean model time {np.mean(ms_hist):.1f} ms, "
              f"mean loop {np.mean(fps_hist):.1f} FPS")


if __name__ == "__main__":
    main()
