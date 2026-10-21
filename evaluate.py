"""Evaluate a trained run: accuracy, macro-F1, top-5, confusion matrix, per-class report,
model size and batch-1 latency.

python evaluate.py --run runs/static_mobilenet_v3_large_static_random_seed0
python evaluate.py --run runs/<run> --split-csv splits/other.csv --data-root data/x --tag external
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slr.engine import measure_latency, pick_device
from slr.evaluation import evaluate_split
from slr.metrics import save_json
from slr.models import count_parameters, load_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory containing best.pt")
    p.add_argument("--split-csv", default=None, help="default: the CSV used for training")
    p.add_argument("--data-root", default=None, help="default: the root used for training")
    p.add_argument("--split", default="test")
    p.add_argument("--tag", default=None, help="prefix for output files (default: split name)")
    p.add_argument("--device", default="auto")
    p.add_argument("--latency-device", default="cpu", help="device for the latency benchmark")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for latency (0 = torch default)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)

    run = Path(a.run)
    device = pick_device(a.device)
    model, ckpt = load_checkpoint(run / "best.pt", device)
    ta = ckpt["train_args"]
    split_csv = a.split_csv or ta["split_csv"]
    data_root = a.data_root or ta["data_root"]

    m = evaluate_split(model, ckpt, split_csv, data_root, a.split, device, out_dir=run, tag=a.tag,
                       batch_size=a.batch_size, workers=a.workers)

    # efficiency (Experiment 2 / 7)
    if a.threads:
        torch.set_num_threads(a.threads)
    lat_dev = torch.device(a.latency_device)
    model.to(lat_dev)
    shape = (3, ta["img_size"], ta["img_size"])
    if ta["task"] == "dynamic":
        shape = (ta["seq_len"], *shape)
    eff = {**count_parameters(model), **measure_latency(model, shape, lat_dev)}
    save_json(eff, run / "efficiency.json")

    print(json.dumps({k: v for k, v in m.items() if k != "most_confused"}, indent=2))
    print("Most confused:", m["most_confused"][:5])
    print("Efficiency:", json.dumps(eff, indent=2))


if __name__ == "__main__":
    main()
