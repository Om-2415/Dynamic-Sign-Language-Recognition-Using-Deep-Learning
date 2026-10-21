"""Dataset indexing, leakage-safe splitting and leakage checks.

Torch-free on purpose so it can be tested and used on any machine.

Metadata format (one row per sample, CSV):
    path    relative path to an image (static) or a frame folder / clip id (dynamic)
    label   class name (string)
    signer  signer / user id (string), or "unknown"
    split   train | val | test   (added by make_splits)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def scan_folder(root: str | Path, layout: str = "class", signer_regex: str | None = None,
                exts: set[str] = IMG_EXT) -> pd.DataFrame:
    """Index a dataset folder.

    layout="class":         root/<class>/<file>
    layout="signer/class":  root/<signer>/<class>/<file>
    signer_regex:           optional regex with one group, applied to the file name
                            (e.g. r"(signer\\d+)" for AUTSL-style names).
    """
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if layout == "class":
            if len(parts) < 2:
                continue
            label, signer = parts[0], "unknown"
        elif layout == "signer/class":
            if len(parts) < 3:
                continue
            signer, label = parts[0], parts[1]
        else:
            raise ValueError(f"unknown layout {layout!r}")
        if signer_regex:
            m = re.search(signer_regex, p.name)
            if m:
                signer = m.group(1)
        rows.append({"path": rel.as_posix(), "label": label, "signer": signer})
    if not rows:
        raise FileNotFoundError(f"No files with extensions {sorted(exts)} found under {root}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _stratified_random(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype=object)
    for _, idx in df.groupby("label").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 3:  # keep at least one training sample per class
            n_test, n_val = max(n_test, 1), max(n_val, 1)
            if n_test + n_val >= n:
                n_test, n_val = 1, 1
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def _signer_independent(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int,
                        test_signers: list[str] | None = None,
                        val_signers: list[str] | None = None) -> pd.Series:
    signers = sorted(df["signer"].astype(str).unique())
    if len(signers) < 3:
        raise ValueError(
            f"Signer-independent split needs >= 3 signers, found {len(signers)} ({signers}). "
            "This dataset cannot support an unseen-user evaluation; use --mode random and "
            "state this limitation in the report.")
    rng = np.random.default_rng(seed)
    order = list(signers)
    rng.shuffle(order)
    if test_signers is None:
        n_test = max(1, int(round(len(signers) * test_frac)))
        test_signers = order[:n_test]
    if val_signers is None:
        rest = [s for s in order if s not in set(test_signers)]
        n_val = max(1, int(round(len(signers) * val_frac)))
        val_signers = rest[:n_val]
    test_set, val_set = set(map(str, test_signers)), set(map(str, val_signers))
    if test_set & val_set:
        raise ValueError("A signer cannot be in both val and test")
    s = df["signer"].astype(str)
    split = pd.Series("train", index=df.index, dtype=object)
    split[s.isin(val_set)] = "val"
    split[s.isin(test_set)] = "test"
    return split


def make_splits(df: pd.DataFrame, mode: str = "random", val_frac: float = 0.15,
                test_frac: float = 0.15, seed: int = 42, **kw) -> pd.DataFrame:
    """Return a copy of df with a `split` column.

    mode="random":      stratified by label (the usual benchmark split)
    mode="signer":      no signer appears in more than one split (unseen-user test)
    mode="predefined":  keep an existing `split` column (e.g. official AUTSL split)
    """
    out = df.copy()
    if mode == "random":
        out["split"] = _stratified_random(out, val_frac, test_frac, seed)
    elif mode == "signer":
        out["split"] = _signer_independent(out, val_frac, test_frac, seed, **kw)
    elif mode == "predefined":
        if "split" not in out:
            raise ValueError("mode=predefined requires a `split` column")
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    return out


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #
def split_report(df: pd.DataFrame) -> dict:
    """Summary + warnings about the split protocol (goes into the report)."""
    rep: dict = {"n_samples": int(len(df)), "n_classes": int(df["label"].nunique()),
                 "n_signers": int(df["signer"].nunique()), "warnings": []}
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        rep[sp] = {"samples": int(len(sub)), "classes": int(sub["label"].nunique()),
                   "signers": sorted(map(str, sub["signer"].unique()))}
    tr = set(df.loc[df.split == "train", "signer"].astype(str))
    te = set(df.loc[df.split == "test", "signer"].astype(str))
    overlap = sorted((tr & te) - {"unknown"})
    rep["signer_overlap_train_test"] = overlap
    rep["signer_independent"] = bool(not overlap and "unknown" not in te and len(te) > 0)
    if rep["n_signers"] <= 1:
        rep["warnings"].append(
            "Only one (or unknown) signer: results measure memorisation of one person's hand, "
            "not generalisation to new users.")
    missing = set(df.label) - set(df.loc[df.split == "train", "label"])
    if missing:
        rep["warnings"].append(f"Classes missing from train: {sorted(missing)}")
    if df["path"].duplicated().any():
        rep["warnings"].append("Duplicate paths in metadata")
    return rep


def dhash(img_gray: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash of a grayscale image (robust to small changes)."""
    import cv2
    small = cv2.resize(img_gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def near_duplicate_rate(train_paths: list[str], test_paths: list[str], root: str | Path,
                        max_test: int = 1000, max_train: int = 20000, threshold: int = 4,
                        seed: int = 0) -> dict:
    """Fraction of test images whose dHash is within `threshold` bits of some train image.

    A high rate (e.g. > 20%) means consecutive video frames of the same take ended up in
    both train and test, so test accuracy is inflated.
    """
    import cv2
    rng = np.random.default_rng(seed)
    root = Path(root)

    def hashes(paths, cap):
        paths = list(paths)
        if len(paths) > cap:
            paths = list(rng.choice(paths, cap, replace=False))
        hs = []
        for p in paths:
            im = cv2.imread(str(root / p), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                hs.append(dhash(im))
        return np.array(hs, dtype=np.uint64)

    h_tr, h_te = hashes(train_paths, max_train), hashes(test_paths, max_test)
    if len(h_tr) == 0 or len(h_te) == 0:
        return {"near_duplicate_rate": float("nan"), "checked_test": int(len(h_te))}
    dup = 0
    for h in h_te:
        dist = np.bitwise_count(np.bitwise_xor(h_tr, h))
        dup += int(dist.min() <= threshold)
    return {"near_duplicate_rate": dup / len(h_te), "checked_test": int(len(h_te)),
            "checked_train": int(len(h_tr)), "hamming_threshold": threshold}
