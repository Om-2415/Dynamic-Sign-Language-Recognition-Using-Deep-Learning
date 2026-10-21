"""Build a test-only split CSV for an external image folder (e.g. Kaggle danrasband/asl-alphabet-test).

python scripts/make_external_csv.py
python scripts/make_external_csv.py --root data/asl_test_external --train-csv splits/static_random.csv --out splits/external_test.csv

Finds class folders at any depth, maps folder names to the training classes
(case-insensitive, 'delete' -> 'del'), and skips classes the model was not trained on (J, Z).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ALIASES = {"delete": "del"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/asl_test_external")
    p.add_argument("--train-csv", default="splits/static_random.csv")
    p.add_argument("--out", default="splits/external_test.csv")
    a = p.parse_args()

    root = Path(a.root)
    if not root.exists():
        sys.exit(f"{root} does not exist. Unzip the external test set there first.")
    train_classes = set(pd.read_csv(a.train_csv)["label"].astype(str))
    lookup = {c.lower(): c for c in train_classes}
    rows, skipped, seen, dupes = [], {}, set(), 0
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        digest = hashlib.md5(f.read_bytes()).hexdigest()  # Kaggle zips often contain the data twice
        if digest in seen:
            dupes += 1
            continue
        seen.add(digest)
        name = f.parent.name.lower()
        label = lookup.get(ALIASES.get(name, name))
        if label is None:
            skipped[f.parent.name] = skipped.get(f.parent.name, 0) + 1
            continue
        rows.append({"path": f.relative_to(root).as_posix(), "label": label,
                     "signer": "external", "split": "test"})
    if not rows:
        sys.exit(f"No images in {root} matched the training classes. Folders seen: {sorted(skipped)[:10]}")
    df = pd.DataFrame(rows)
    if len(df) < 100:
        print(f"WARNING: only {len(df)} images. The Kaggle 'asl_alphabet_test' folder from the TRAINING zip has "
              "just 1 image per class; you want the separate danrasband/asl-alphabet-test download (~30 per class).")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images, {df.label.nunique()} classes -> {a.out}")
    if dupes:
        print(f"Removed {dupes} exact duplicate image files (same content found twice in the download).")
    if skipped:
        print("Skipped folders (not in training classes):", skipped)
    print(df.label.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
