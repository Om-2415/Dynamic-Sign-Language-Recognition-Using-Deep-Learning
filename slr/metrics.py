"""Evaluation metrics and plots (torch-free)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def compute_metrics(y_true, y_pred, class_names: list[str], probs: np.ndarray | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    m = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    if probs is not None:
        m["top5_accuracy"] = topk_accuracy(probs, y_true, 5)
    m["most_confused"] = most_confused_pairs(y_true, y_pred, class_names, top=10)
    return m


def most_confused_pairs(y_true, y_pred, class_names, top: int = 10) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[:top]
    pairs = []
    for f in flat:
        i, j = np.unravel_index(f, cm.shape)
        if cm[i, j] == 0:
            break
        pairs.append({"true": class_names[i], "predicted": class_names[j], "count": int(cm[i, j])})
    return pairs


def per_class_report(y_true, y_pred, class_names) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, output_dict=True, zero_division=0)
    rows = [{"class": c, **rep[c]} for c in class_names]
    return pd.DataFrame(rows).rename(columns={"f1-score": "f1"})


def save_confusion_matrix(y_true, y_pred, class_names, path: str | Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).astype(float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    size = min(max(6, n * 0.35), 30)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ticks = np.arange(n)
    show_labels = n <= 60
    ax.set_xticks(ticks if show_labels else [])
    ax.set_yticks(ticks if show_labels else [])
    if show_labels:
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
