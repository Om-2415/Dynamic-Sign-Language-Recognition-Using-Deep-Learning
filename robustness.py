"""Experiment 3: robustness to controlled image variation.

Evaluates a trained run on the clean test set and on perturbed copies of it
(brightness, contrast, rotation, scale, translation, blur, noise, occlusion,
low resolution, and optionally background replacement) at severities 1-3.

python robustness.py --run runs/<run>
python robustness.py --run runs/<run> --perturbations brightness rotation --severities 1 2 3
python robustness.py --run runs/<run> --mask-root data/masks --background-dir data/backgrounds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from slr.engine import pick_device
from slr.evaluation import evaluate_split
from slr.models import load_checkpoint
from slr.perturb import PERTURBATIONS, BackgroundReplacer, DatasetPerturbation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--split-csv", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS))
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--mask-root", default=None, help="foreground masks for the background test")
    p.add_argument("--background-dir", default=None, help="background images for the background test")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)

    run = Path(a.run)
    device = pick_device(a.device)
    model, ckpt = load_checkpoint(run / "best.pt", device)
    ta = ckpt["train_args"]
    split_csv, root = a.split_csv or ta["split_csv"], a.data_root or ta["data_root"]
    kw = dict(device=device, batch_size=a.batch_size, workers=a.workers, save_artifacts=False)

    tests: dict[str, object] = {n: PERTURBATIONS[n] for n in a.perturbations}
    if a.mask_root and a.background_dir:
        tests["background"] = BackgroundReplacer(a.mask_root, a.background_dir)
    elif a.mask_root or a.background_dir:
        raise SystemExit("--mask-root and --background-dir must be given together")

    clean = evaluate_split(model, ckpt, split_csv, root, a.split, **kw)
    rows = [{"perturbation": "clean", "severity": 0, "accuracy": clean["accuracy"],
             "macro_f1": clean["macro_f1"], "acc_drop": 0.0, "f1_drop": 0.0}]
    print(f"clean          acc {clean['accuracy']:.4f}  macro-F1 {clean['macro_f1']:.4f}")

    for name, fn in tests.items():
        for sev in a.severities:
            perturb = DatasetPerturbation(fn, sev, needs_path=(name == "background"))
            m = evaluate_split(model, ckpt, split_csv, root, a.split, perturb=perturb, **kw)
            rows.append({"perturbation": name, "severity": sev, "accuracy": m["accuracy"],
                         "macro_f1": m["macro_f1"], "acc_drop": clean["accuracy"] - m["accuracy"],
                         "f1_drop": clean["macro_f1"] - m["macro_f1"]})
            print(f"{name:14s} s{sev}  acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
                  f"drop {rows[-1]['acc_drop']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run / "robustness.csv", index=False)
    plot(df, run / "robustness.png", ckpt["run_name"])
    worst = df[df.severity > 0].groupby("perturbation")["acc_drop"].max().sort_values(ascending=False)
    print("\nLargest accuracy drop per perturbation:\n" + worst.to_string())
    print(f"\nSaved {run / 'robustness.csv'} and robustness.png")


def plot(df: pd.DataFrame, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clean = df.loc[df.perturbation == "clean", "accuracy"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in df[df.severity > 0].groupby("perturbation"):
        g = g.sort_values("severity")
        ax.plot([0, *g.severity], [clean, *g.accuracy], marker="o", label=name)
    ax.set_xlabel("severity (0 = clean)")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(sorted(df.severity.unique()))
    ax.set_title(f"Robustness - {title}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
