"""Training / inference loops and latency measurement."""
from __future__ import annotations

import contextlib
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, amp: bool = False,
                    grad_clip: float = 1.0, log_every: int = 0) -> dict:
    model.train()
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    tot_loss, correct, n = 0.0, 0, 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with (torch.autocast("cuda") if use_amp else contextlib.nullcontext()):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        tot_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        if log_every and step % log_every == 0:
            print(f"    step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)
    return {"loss": tot_loss / max(n, 1), "accuracy": correct / max(n, 1)}


@torch.no_grad()
def predict(model, loader, device, criterion=None) -> dict:
    """Returns y_true, y_pred, probs (numpy) and the mean loss if a criterion is given."""
    model.eval()
    ys, ps, tot_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            tot_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        ys.append(y.cpu())
        ps.append(torch.softmax(logits.float(), 1).cpu())
    probs = torch.cat(ps).numpy()
    y_true = torch.cat(ys).numpy()
    return {"y_true": y_true, "y_pred": probs.argmax(1), "probs": probs,
            "loss": tot_loss / max(n, 1) if criterion is not None else None}


@torch.no_grad()
def measure_latency(model, input_shape: tuple, device, warmup: int = 10, runs: int = 50) -> dict:
    """Batch-1 inference latency, the number that matters for a webcam app."""
    model.eval()
    x = torch.randn(1, *input_shape, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"device": str(device), "latency_ms_median": float(np.median(t)),
            "latency_ms_p95": float(np.percentile(t, 95)),
            "throughput_fps": float(1000 / np.median(t)), "threads": torch.get_num_threads()}
