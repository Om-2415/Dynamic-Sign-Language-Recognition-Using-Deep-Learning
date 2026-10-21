"""Train a static or dynamic sign recognition model.

Examples
--------
# Experiment 1: custom CNN baseline (from scratch)
python train.py --task static --arch custom_cnn --data-root data/asl_alphabet \
    --split-csv splits/static_random.csv --epochs 30 --lr 1e-3

# Experiment 2: transfer learning (frozen backbone for 2 epochs, then fine-tune)
python train.py --task static --arch mobilenet_v3_large --data-root data/asl_alphabet \
    --split-csv splits/static_random.csv --epochs 15 --freeze-epochs 2

# Experiment 5: CNN + LSTM on frame sequences
python train.py --task dynamic --arch mobilenet_v3_small --rnn lstm --seq-len 16 \
    --data-root data/frames --split-csv splits/dynamic_signer.csv --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slr.data import class_list, make_dataset
from slr.engine import (pick_device, predict, set_seed, train_one_epoch, warmup_cosine,
                        worker_init_fn)
from slr.evaluation import evaluate_split
from slr.metrics import compute_metrics, save_json
from slr.models import RNN_TYPES, STATIC_ARCHS, build_model, count_parameters


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["static", "dynamic"], required=True)
    p.add_argument("--arch", choices=STATIC_ARCHS, default="mobilenet_v3_small",
                   help="image model, or per-frame feature extractor for --task dynamic")
    p.add_argument("--data-root", required=True, help="image root (static) or frames root (dynamic)")
    p.add_argument("--split-csv", required=True, help="CSV with path,label,signer,split")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None, help="run name; default derived from settings")
    # optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default 1e-3 (scratch) / 3e-4 (pretrained)")
    p.add_argument("--backbone-lr-mult", type=float, default=None,
                   help="lr multiplier for pretrained backbone (default 0.1, 1.0 for custom_cnn)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=7, help="early stopping on val macro-F1")
    p.add_argument("--freeze-epochs", type=int, default=0, help="train only the head for N epochs first")
    p.add_argument("--freeze-backbone", action="store_true", help="never fine-tune the backbone")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)
    # data
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--aug", choices=["standard", "none"], default="standard")
    p.add_argument("--hflip", action="store_true", help="horizontal-flip augmentation (off by default)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="use only N samples per split (debugging)")
    # dynamic
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--rnn", choices=RNN_TYPES, default="lstm")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--pool", choices=["last", "mean"], default="last")
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--no-test", action="store_true", help="skip test evaluation at the end")
    return p.parse_args(argv)


def default_name(a) -> str:
    split = Path(a.split_csv).stem
    if a.task == "static":
        core = a.arch
    else:
        core = f"{a.arch}+{a.rnn}" + ("_bi" if a.bidirectional else "") + (
            "_frozen" if a.freeze_backbone else "")
    return f"{a.task}_{core}_{split}"


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    device = pick_device(a.device)
    pretrained = not a.no_pretrained and a.arch != "custom_cnn"
    lr = a.lr or (3e-4 if pretrained else 1e-3)
    bb_mult = a.backbone_lr_mult if a.backbone_lr_mult is not None else (0.1 if pretrained else 1.0)
    experiment = a.name or default_name(a)
    run_name = f"{experiment}_seed{a.seed}"
    out = Path(a.out) / run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}  device={device}  pretrained={pretrained}")

    df = pd.read_csv(a.split_csv)
    df["label"] = df["label"].astype(str)
    classes = class_list(df)
    splits = {s: df[df.split == s] for s in ("train", "val", "test")}
    if a.limit:
        splits = {s: d.sample(min(a.limit, len(d)), random_state=a.seed) for s, d in splits.items()}
    if splits["val"].empty:
        raise SystemExit("Split CSV has no 'val' rows; model selection needs a validation split.")
    print({s: len(d) for s, d in splits.items()}, f"classes={len(classes)}")

    common = dict(task=a.task, root=a.data_root, classes=classes, img_size=a.img_size, seq_len=a.seq_len)
    ds_train = make_dataset(df=splits["train"], train=True, aug=a.aug, hflip=a.hflip, **common)
    ds_val = make_dataset(df=splits["val"], train=False, **common)
    g = torch.Generator().manual_seed(a.seed)
    dl_train = DataLoader(ds_train, a.batch_size, shuffle=True, num_workers=a.workers, generator=g,
                          worker_init_fn=worker_init_fn, pin_memory=device.type == "cuda",
                          persistent_workers=a.workers > 0)
    dl_val = DataLoader(ds_val, a.batch_size, shuffle=False, num_workers=a.workers)

    model_cfg = {"task": a.task, "arch": a.arch, "num_classes": len(classes), "pretrained": pretrained,
                 "dropout": a.dropout, "rnn": a.rnn, "hidden": a.hidden, "layers": a.layers,
                 "bidirectional": a.bidirectional, "pool": a.pool}
    model = build_model(model_cfg).to(device)
    sizes = count_parameters(model)
    print(f"Model params: {sizes['params_millions']} M ({sizes['size_mb']} MB)")

    bb_params = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    other = [p for p in model.parameters() if id(p) not in bb_ids]
    optimizer = torch.optim.AdamW([{"params": bb_params, "lr": lr * bb_mult},
                                   {"params": other, "lr": lr}], weight_decay=a.weight_decay)
    steps = a.epochs * max(1, len(dl_train))
    scheduler = warmup_cosine(optimizer, int(a.warmup_epochs * len(dl_train)), steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    train_args = vars(a) | {"lr": lr, "backbone_lr_mult": bb_mult, "pretrained": pretrained,
                            "experiment": experiment, "device": str(device)}
    save_json(train_args, out / "config.json")

    best_f1, best_epoch, history = -1.0, -1, []
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        frozen = a.freeze_backbone or epoch <= a.freeze_epochs
        model.set_backbone_trainable(not frozen)
        t0 = time.time()
        tr = train_one_epoch(model, dl_train, optimizer, scheduler, criterion, device, amp=a.amp)
        val = predict(model, dl_val, device, criterion)
        vm = compute_metrics(val["y_true"], val["y_pred"], classes)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
               "val_loss": val["loss"], "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
               "backbone_frozen": frozen, "seconds": time.time() - t0}
        history.append(row)
        flag = ""
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, flag = vm["macro_f1"], epoch, "  *best*"
            torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "classes": classes,
                        "train_args": train_args, "run_name": run_name, "epoch": epoch,
                        "val_metrics": vm}, out / "best.pt")
        print(f"[{epoch:3d}/{a.epochs}] train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f} | "
              f"val loss {val['loss']:.4f} acc {vm['accuracy']:.3f} macro-F1 {vm['macro_f1']:.3f} "
              f"| {row['seconds']:.0f}s{flag}", flush=True)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if epoch - best_epoch >= a.patience:
            print(f"Early stopping: no val macro-F1 improvement for {a.patience} epochs.")
            break

    plot_history(pd.DataFrame(history), out / "training_curves.png", run_name)
    summary = {"run_name": run_name, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "train_minutes": (time.time() - t_start) / 60, **sizes}
    save_json(summary, out / "train_summary.json")
    print(json.dumps(summary, indent=2))

    if not a.no_test and not splits["test"].empty:
        from slr.models import load_checkpoint
        best, ckpt = load_checkpoint(out / "best.pt", device)
        csv = a.split_csv
        if a.limit:  # evaluate on the same limited subset
            csv = out / "limited_split.csv"
            pd.concat(splits.values()).to_csv(csv, index=False)
        m = evaluate_split(best, ckpt, csv, a.data_root, "test", device, out_dir=out,
                           workers=a.workers, batch_size=a.batch_size)
        print(f"TEST  accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")
    return out


def plot_history(h: pd.DataFrame, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.train_acc, label="train acc")
    ax[1].plot(h.epoch, h.val_acc, label="val acc")
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1", linestyle="--")
    ax[1].set_title("Accuracy / macro-F1")
    for x in ax:
        x.set_xlabel("epoch")
        x.legend()
        x.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
