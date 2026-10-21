"""Prepare a video sign dataset for the dynamic (CNN+LSTM/GRU) experiments.

Extracts a fixed number of uniformly spaced frames per clip into
<frames-root>/<clip_id>/00000.jpg ... and writes a split CSV (path = clip_id).

Sources (pick one)
  --layout class|signer/class   videos-root/<class>/<video>  or  videos-root/<signer>/<class>/<video>
  --metadata FILE.csv           columns: video (relative to videos-root), label, signer[, split]
  --wlasl-json WLASL_v0.3.json  WLASL; videos-root holds <video_id>.mp4; use --top-k 100 for WLASL-100
  --autsl-labels train=a.csv val=b.csv test=c.csv
                                AUTSL label files (rows: sample,class_id); videos-root holds
                                <sample>_color.mp4. Optional --autsl-classes SignList_ClassId_TR_EN.csv

Split
  --mode predefined   keep the dataset's official split (WLASL / AUTSL; AUTSL's is signer-independent)
  --mode random | signer   make a new split (see prepare_static.py)

Example
  python prepare_video.py --wlasl-json data/WLASL_v0.3.json --videos-root data/wlasl/videos \
      --top-k 100 --frames-root data/frames_wlasl100 --mode predefined --out splits/wlasl100.csv
"""
from __future__ import annotations

import argparse
import json
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from slr.metrics import save_json
from slr.splits import VIDEO_EXT, make_splits, scan_folder, split_report
from slr.video import extract_frames


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def from_folder(videos_root, layout, signer_regex) -> pd.DataFrame:
    df = scan_folder(videos_root, layout, signer_regex, exts=VIDEO_EXT).rename(columns={"path": "video"})
    df["clip_id"] = [_safe(f"{l}__{Path(v).stem}") for l, v in zip(df.label, df.video)]
    return df


def from_metadata(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"video", "label"}
    if not need <= set(df.columns):
        raise SystemExit(f"--metadata needs columns {need}, found {list(df.columns)}")
    if "signer" not in df:
        df["signer"] = "unknown"
    df["clip_id"] = [_safe(Path(v).with_suffix("").as_posix()) for v in df.video]
    return df


def from_wlasl(json_path, videos_root, top_k: int, use_frame_range: bool) -> pd.DataFrame:
    """WLASL JSON is ordered by gloss frequency, so the first K glosses form WLASL-K."""
    entries = json.load(open(json_path))[:top_k] if top_k else json.load(open(json_path))
    rows, missing = [], 0
    for e in entries:
        for inst in e["instances"]:
            vid = f"{inst['video_id']}.mp4"
            if not (Path(videos_root) / vid).exists():
                missing += 1
                continue
            r = {"video": vid, "label": e["gloss"], "signer": str(inst.get("signer_id", "unknown")),
                 "split": inst.get("split", "train"), "clip_id": str(inst["video_id"])}
            if use_frame_range:  # 1-based start, -1 = until end
                r["start"] = max(int(inst.get("frame_start", 1)) - 1, 0)
                r["end"] = int(inst.get("frame_end", -1))
            rows.append(r)
    total = len(rows) + missing
    print(f"WLASL: {len(entries)} glosses, {len(rows)}/{total} videos found locally "
          f"({missing} missing - many WLASL source URLs are dead; report the number you used).")
    return pd.DataFrame(rows)


def from_autsl(label_specs: list[str], videos_root, classes_csv=None) -> pd.DataFrame:
    """Verify column layout against your AUTSL download; adjust here if it differs."""
    names = None
    if classes_csv:
        c = pd.read_csv(classes_csv)
        id_col = c.columns[0]
        en_col = "EN" if "EN" in c.columns else c.columns[-1]
        names = dict(zip(c[id_col].astype(int), c[en_col].astype(str)))
    index = {p.name: p for p in Path(videos_root).rglob("*_color.mp4")}
    rows, missing = [], 0
    for spec in label_specs:
        split, path = spec.split("=", 1)
        lab = pd.read_csv(path, header=None, names=["sample", "class_id"])
        for s, cid in zip(lab["sample"], lab["class_id"]):
            f = index.get(f"{s}_color.mp4")
            if f is None:
                missing += 1
                continue
            m = re.search(r"(signer\d+)", s)
            rows.append({"video": f.relative_to(videos_root).as_posix(),
                         "label": names.get(int(cid), str(cid)) if names else str(cid),
                         "signer": m.group(1) if m else "unknown", "split": split, "clip_id": _safe(s)})
    print(f"AUTSL: {len(rows)} clips found, {missing} listed but missing")
    return pd.DataFrame(rows)


def _extract(row: dict, videos_root, frames_root, num_frames, short_side) -> int:
    out = Path(frames_root) / row["clip_id"]
    if out.exists() and len(list(out.glob("*.jpg"))) >= num_frames:
        return num_frames  # already done (resumable)
    start = row.get("start")
    end = row.get("end")
    start = None if pd.isna(start) else int(start)
    end = None if pd.isna(end) else int(end)
    try:
        return extract_frames(Path(videos_root) / row["video"], out, num_frames, short_side, start, end)
    except Exception as e:  # corrupt video: skip it, report count
        print(f"  failed {row['video']}: {e}")
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-root", required=True)
    p.add_argument("--frames-root", required=True)
    p.add_argument("--out", required=True, help="output split CSV")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--layout", choices=["class", "signer/class"])
    src.add_argument("--metadata")
    src.add_argument("--wlasl-json")
    src.add_argument("--autsl-labels", nargs="+")
    p.add_argument("--signer-regex", default=None)
    p.add_argument("--top-k", type=int, default=100, help="WLASL: number of glosses (0 = all 2000)")
    p.add_argument("--wlasl-frame-range", action="store_true",
                   help="WLASL: trim to frame_start/frame_end (use if your videos are untrimmed)")
    p.add_argument("--autsl-classes", default=None)
    p.add_argument("--mode", choices=["predefined", "random", "signer"], default="predefined")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-frames", type=int, default=32, help="frames stored per clip")
    p.add_argument("--short-side", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)

    if a.wlasl_json:
        df = from_wlasl(a.wlasl_json, a.videos_root, a.top_k, a.wlasl_frame_range)
    elif a.autsl_labels:
        df = from_autsl(a.autsl_labels, a.videos_root, a.autsl_classes)
    elif a.metadata:
        df = from_metadata(a.metadata)
    else:
        df = from_folder(a.videos_root, a.layout or "class", a.signer_regex)
    if df.empty:
        raise SystemExit("No videos found.")
    if df["clip_id"].duplicated().any():
        raise SystemExit("Duplicate clip ids; make video names unique.")

    fn = partial(_extract, videos_root=a.videos_root, frames_root=a.frames_root,
                 num_frames=a.num_frames, short_side=a.short_side)
    records = df.to_dict("records")
    if a.workers > 1:
        with Pool(a.workers) as pool:
            counts = []
            for i, c in enumerate(pool.imap(fn, records, chunksize=4), 1):
                counts.append(c)
                if i % 200 == 0:
                    print(f"  {i}/{len(records)} clips", flush=True)
    else:
        counts = [fn(r) for r in records]
    df["n_frames"] = counts
    failed = int((df.n_frames == 0).sum())
    df = df[df.n_frames > 0].copy()
    df["path"] = df["clip_id"]

    if a.mode == "predefined" and "split" not in df:
        raise SystemExit("This source has no official split; use --mode random or --mode signer.")
    if a.mode != "predefined":
        df = df.drop(columns=["split"], errors="ignore")
    df = make_splits(df, a.mode, a.val_frac, a.test_frac, a.seed)
    # labels that never occur in train cannot be learned; drop them from val/test with a note
    train_labels = set(df.loc[df.split == "train", "label"])
    dropped = df[~df.label.isin(train_labels)]
    df = df[df.label.isin(train_labels)]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "signer", "split", "video", "n_frames"]
    df[cols].to_csv(out, index=False)
    rep = split_report(df) | {"mode": a.mode, "failed_videos": failed,
                              "dropped_not_in_train": int(len(dropped))}
    save_json(rep, out.with_suffix(".report.json"))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("train", "val", "test")}, indent=2))
    for s in ("train", "val", "test"):
        print(f"{s:5s}: {rep[s]['samples']:6d} clips, {rep[s]['classes']} classes, "
              f"{len(rep[s]['signers'])} signers")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
