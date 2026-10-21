"""Index a static sign-image dataset and write a leakage-checked split CSV.

Folder layouts
  --layout class          root/<class>/<image>          (e.g. Kaggle ASL Alphabet)
  --layout signer/class   root/<signer>/<class>/<image> (multi-signer datasets)
  --signer-regex          take the signer id from the file name instead

Examples
  python prepare_static.py --root data/asl_alphabet --mode random --out splits/static_random.csv --check-duplicates
  python prepare_static.py --root data/multi_signer --layout signer/class --mode signer \
      --out splits/static_signer.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from slr.metrics import save_json
from slr.splits import make_splits, near_duplicate_rate, scan_folder, split_report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--layout", choices=["class", "signer/class"], default="class")
    p.add_argument("--signer-regex", default=None)
    p.add_argument("--mode", choices=["random", "signer"], default="random")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--test-signers", nargs="*", default=None, help="explicit held-out test signers")
    p.add_argument("--val-signers", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-classes", nargs="*", default=[],
                   help="e.g. J Z (motion letters) or 'nothing' 'space' 'del'")
    p.add_argument("--check-duplicates", action="store_true",
                   help="estimate near-duplicate leakage between train and test (perceptual hash)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    df = scan_folder(a.root, a.layout, a.signer_regex)
    if a.exclude_classes:
        df = df[~df.label.isin(a.exclude_classes)]
    kw = {}
    if a.mode == "signer":
        kw = {"test_signers": a.test_signers, "val_signers": a.val_signers}
    df = make_splits(df, a.mode, a.val_frac, a.test_frac, a.seed, **kw)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rep = split_report(df) | {"mode": a.mode, "root": str(a.root), "seed": a.seed}
    if a.check_duplicates:
        rep["duplicates"] = near_duplicate_rate(df.loc[df.split == "train", "path"].tolist(),
                                                df.loc[df.split == "test", "path"].tolist(), a.root)
        r = rep["duplicates"]["near_duplicate_rate"]
        if r == r and r > 0.2:
            rep["warnings"].append(
                f"{r:.0%} of checked test images are near-duplicates of training images: "
                "test accuracy on this split will be inflated.")
    save_json(rep, out.with_suffix(".report.json"))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("train", "val", "test")}, indent=2))
    for s in ("train", "val", "test"):
        print(f"{s:5s}: {rep[s]['samples']:6d} samples, {rep[s]['classes']} classes, "
              f"{len(rep[s]['signers'])} signers")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
