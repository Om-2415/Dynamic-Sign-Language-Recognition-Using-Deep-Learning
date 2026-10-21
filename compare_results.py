"""Collect all runs into one results table (mean +/- std over seeds) and compute the
generalisation gap between random and signer-independent splits.

python compare_results.py --runs runs --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from slr.metrics import load_json


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(runs_dir.iterdir()):
        cfg_f, test_f = d / "config.json", d / "test_metrics.json"
        if not (cfg_f.exists() and test_f.exists()):
            continue
        cfg, test = load_json(cfg_f), load_json(test_f)
        model = cfg["arch"] if cfg["task"] == "static" else f"{cfg['arch']}+{cfg['rnn']}"
        row = {"experiment": cfg["experiment"], "task": cfg["task"], "model": model,
               "split": Path(cfg["split_csv"]).stem, "seed": cfg["seed"],
               "accuracy": test["accuracy"], "macro_f1": test["macro_f1"],
               "top5": test.get("top5_accuracy")}
        if (d / "efficiency.json").exists():
            eff = load_json(d / "efficiency.json")
            row |= {"params_M": eff["params_millions"], "size_MB": eff["size_mb"],
                    "latency_ms": eff["latency_ms_median"], "fps": eff["throughput_fps"]}
        if (d / "robustness.csv").exists():
            rob = pd.read_csv(d / "robustness.csv")
            row["mean_robust_acc_drop"] = rob.loc[rob.severity > 0, "acc_drop"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean, std, n):
    return f"{mean:.3f} ± {std:.3f}" if n > 1 else f"{mean:.3f}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = collect(Path(a.runs))
    if df.empty:
        raise SystemExit(f"No evaluated runs found in {a.runs}")
    df.to_csv(out / "all_runs.csv", index=False)

    num = [c for c in ["accuracy", "macro_f1", "top5", "params_M", "size_MB", "latency_ms", "fps",
                       "mean_robust_acc_drop"] if c in df]
    g = df.groupby(["task", "model", "split"])
    agg = g[num].agg(["mean", "std"])
    agg["n_seeds"] = g.size()
    agg.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "summary.csv", index=False)

    lines = ["| Task | Model | Split | Seeds | Accuracy | Macro-F1 | Params (M) | Latency (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        n = r.n_seeds
        lat = f"{getattr(r, 'latency_ms_mean'):.1f}" if "latency_ms_mean" in agg and pd.notna(r.latency_ms_mean) else "-"
        par = f"{getattr(r, 'params_M_mean'):.2f}" if "params_M_mean" in agg and pd.notna(r.params_M_mean) else "-"
        lines.append(f"| {r.task} | {r.model} | {r.split} | {n} | "
                     f"{fmt(r.accuracy_mean, r.accuracy_std or 0, n)} | "
                     f"{fmt(r.macro_f1_mean, r.macro_f1_std or 0, n)} | {par} | {lat} |")

    # generalisation gap: same model, split names containing 'random' vs 'signer'
    gaps = []
    for (task, model), sub in agg.groupby(["task", "model"]):
        rnd = sub[sub.split.str.contains("random")]
        sig = sub[sub.split.str.contains("signer")]
        if len(rnd) and len(sig):
            gaps.append({"task": task, "model": model,
                         "random_acc": rnd.accuracy_mean.iloc[0], "signer_acc": sig.accuracy_mean.iloc[0],
                         "gap": rnd.accuracy_mean.iloc[0] - sig.accuracy_mean.iloc[0]})
    if gaps:
        gdf = pd.DataFrame(gaps)
        gdf.to_csv(out / "generalisation_gap.csv", index=False)
        lines += ["", "**Generalisation gap (random-split accuracy − unseen-signer accuracy)**", "",
                  "| Task | Model | Random | Unseen signer | Gap |", "|---|---|---|---|---|"]
        lines += [f"| {r.task} | {r.model} | {r.random_acc:.3f} | {r.signer_acc:.3f} | {r.gap:+.3f} |"
                  for r in gdf.itertuples()]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}/results.md, summary.csv, all_runs.csv")


if __name__ == "__main__":
    main()
