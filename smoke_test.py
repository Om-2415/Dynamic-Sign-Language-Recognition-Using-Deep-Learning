"""End-to-end check of every experiment on tiny synthetic data (about 2-5 min on a laptop CPU).

python smoke_test.py            # keeps outputs in _smoke/
python smoke_test.py --clean    # delete _smoke/ afterwards

Pretrained weights are NOT downloaded here (--no-pretrained) so it also works offline.
The numbers it prints are meaningless; it only proves the code runs on your machine.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W = ROOT / "_smoke"


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n$ " + " ".join(cmd[1:]), flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\nFAILED: {' '.join(map(str, args))}")
    print(f"  ok ({time.time() - t:.0f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()
    if W.exists():
        shutil.rmtree(W)
    data, sp, runs = W / "data", W / "splits", W / "runs"
    fast = ["--epochs", "3", "--img-size", "64", "--batch-size", "16", "--workers", "0", "--patience", "10",
            "--out", runs]

    run("tests/test_core.py")
    run("scripts/make_synthetic_data.py", "--out", data, "--static-per-class", "12", "--videos-per-class", "4")
    run("prepare_static.py", "--root", data / "static", "--layout", "signer/class", "--mode", "random",
        "--out", sp / "static_random.csv", "--check-duplicates")
    run("prepare_static.py", "--root", data / "static", "--layout", "signer/class", "--mode", "signer",
        "--out", sp / "static_signer.csv")
    run("prepare_video.py", "--videos-root", data / "videos", "--layout", "signer/class",
        "--frames-root", data / "frames", "--mode", "signer", "--num-frames", "16", "--workers", "1",
        "--out", sp / "dynamic_signer.csv")

    # Experiments 1, 2, 4: static models on random and unseen-signer splits
    for arch in ("custom_cnn", "mobilenet_v3_small", "efficientnet_b0"):
        run("train.py", "--task", "static", "--arch", arch, "--no-pretrained", "--data-root", data / "static",
            "--split-csv", sp / "static_random.csv", *fast)
    run("train.py", "--task", "static", "--arch", "custom_cnn", "--data-root", data / "static",
        "--split-csv", sp / "static_signer.csv", *fast)
    static_run = runs / "static_custom_cnn_static_random_seed0"
    run("evaluate.py", "--run", static_run, "--workers", "0")
    # Experiment 3
    run("robustness.py", "--run", static_run, "--perturbations", "brightness", "rotation", "occlusion",
        "--severities", "1", "3", "--workers", "0")
    # Experiment 6 (static stream)
    run("stability.py", "--run", static_run, "--num-samples", "10", "--hold", "15")

    # Experiment 5: CNN+LSTM, CNN+GRU, and the no-temporal-model ablation
    for rnn in ("lstm", "gru", "none"):
        run("train.py", "--task", "dynamic", "--arch", "custom_cnn", "--rnn", rnn, "--seq-len", "8",
            "--data-root", data / "frames", "--split-csv", sp / "dynamic_signer.csv", *fast)
    dyn_run = runs / "dynamic_custom_cnn+lstm_dynamic_signer_seed0"
    run("evaluate.py", "--run", dyn_run, "--workers", "0")
    run("stability.py", "--run", dyn_run, "--num-samples", "6")
    # Experiment 7 without a camera: play a synthetic video through the live pipeline
    video = next((data / "videos").rglob("*.mp4"))
    run("webcam.py", "--run", dyn_run, "--source", video, "--no-display", "--stride", "2",
        "--save-video", W / "demo_out.mp4", "--log", W / "demo_log.csv")
    run("webcam.py", "--run", static_run, "--source", video, "--no-display", "--log", W / "demo_static.csv")

    run("compare_results.py", "--runs", runs, "--out", W / "results")
    print("\nSMOKE TEST PASSED - the full pipeline runs on this machine.")
    if a.clean:
        shutil.rmtree(W)


if __name__ == "__main__":
    main()
