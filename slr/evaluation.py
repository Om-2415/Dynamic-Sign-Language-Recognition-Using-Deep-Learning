"""Shared evaluation used by train.py, evaluate.py and robustness.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import make_dataset
from .engine import predict
from .metrics import compute_metrics, per_class_report, save_confusion_matrix, save_json


def build_loader(ckpt: dict, df: pd.DataFrame, data_root, perturb=None, batch_size: int = 64,
                 workers: int = 2) -> DataLoader:
    a = ckpt["train_args"]
    ds = make_dataset(a["task"], df, data_root, ckpt["classes"], a["img_size"], train=False,
                      seq_len=a.get("seq_len", 16), perturb=perturb)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)


def evaluate_split(model, ckpt: dict, split_csv, data_root, split: str = "test", device="cpu",
                   out_dir: str | Path | None = None, tag: str | None = None, perturb=None,
                   batch_size: int = 64, workers: int = 2, save_artifacts: bool = True) -> dict:
    df = pd.read_csv(split_csv)
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"No rows with split == {split!r} in {split_csv}")
    unknown = set(df["label"].astype(str)) - set(ckpt["classes"])
    if unknown:
        raise ValueError(f"Labels not seen in training: {sorted(unknown)[:10]}")
    loader = build_loader(ckpt, df, data_root, perturb, batch_size, workers)
    res = predict(model, loader, torch.device(device) if isinstance(device, str) else device)
    classes = ckpt["classes"]
    m = compute_metrics(res["y_true"], res["y_pred"], classes, res["probs"])
    m.update({"split": split, "split_csv": str(split_csv)})
    if out_dir is not None and save_artifacts:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = tag or split
        save_json(m, out / f"{tag}_metrics.json")
        per_class_report(res["y_true"], res["y_pred"], classes).to_csv(out / f"{tag}_per_class.csv", index=False)
        save_confusion_matrix(res["y_true"], res["y_pred"], classes, out / f"{tag}_confusion_matrix.png",
                              title=f"{ckpt['run_name']} - {tag}")
        pd.DataFrame({"path": df["path"].values, "true": [classes[i] for i in res["y_true"]],
                      "pred": [classes[i] for i in res["y_pred"]],
                      "confidence": res["probs"].max(1)}).to_csv(out / f"{tag}_predictions.csv", index=False)
    return m
