"""Tests for the torch-free core. Run:  python tests/test_core.py   (or pytest)"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slr import perturb, smoothing, video  # noqa: E402
from slr.metrics import compute_metrics, per_class_report, save_confusion_matrix  # noqa: E402
from slr.splits import make_splits, near_duplicate_rate, split_report  # noqa: E402


def toy_meta(n_signers=6, n_classes=5, per=10):
    rows = [{"path": f"s{s}/c{c}/{k}.png", "label": f"c{c}", "signer": f"s{s}"}
            for s in range(n_signers) for c in range(n_classes) for k in range(per)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- splits
def test_random_split_is_stratified_and_complete():
    df = make_splits(toy_meta(), "random", 0.15, 0.15, seed=1)
    assert set(df.split) == {"train", "val", "test"}
    for _, g in df.groupby("label"):
        assert (g.split == "test").sum() == 9 and (g.split == "val").sum() == 9
    assert len(df) == 300


def test_signer_split_has_no_overlap():
    df = make_splits(toy_meta(), "signer", 0.15, 0.15, seed=3)
    s = {sp: set(df[df.split == sp].signer) for sp in ("train", "val", "test")}
    assert not (s["train"] & s["test"]) and not (s["train"] & s["val"]) and not (s["val"] & s["test"])
    rep = split_report(df)
    assert rep["signer_independent"] and rep["signer_overlap_train_test"] == []


def test_signer_split_explicit_and_deterministic():
    a = make_splits(toy_meta(), "signer", seed=5)
    b = make_splits(toy_meta(), "signer", seed=5)
    assert a.split.tolist() == b.split.tolist()
    c = make_splits(toy_meta(), "signer", test_signers=["s0"], val_signers=["s1"])
    assert set(c[c.split == "test"].signer) == {"s0"}


def test_signer_split_refuses_single_signer():
    try:
        make_splits(toy_meta(n_signers=1), "signer")
    except ValueError as e:
        assert "unseen-user" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_random_split_warns_single_signer():
    rep = split_report(make_splits(toy_meta(n_signers=1), "random"))
    assert any("one" in w for w in rep["warnings"])


def test_near_duplicate_detection():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        train, test_dup, test_new = [], [], []
        for i in range(20):
            base = cv2.GaussianBlur(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), (9, 9), 0)
            cv2.imwrite(str(d / f"tr{i}.png"), base)
            train.append(f"tr{i}.png")
            near = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"dup{i}.png"), near)  # like the next video frame
            test_dup.append(f"dup{i}.png")
            other = cv2.GaussianBlur(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), (9, 9), 0)
            cv2.imwrite(str(d / f"new{i}.png"), other)
            test_new.append(f"new{i}.png")
        assert near_duplicate_rate(train, test_dup, d)["near_duplicate_rate"] > 0.9
        assert near_duplicate_rate(train, test_new, d)["near_duplicate_rate"] < 0.1


# ---------------------------------------------------------------- perturbations
def test_perturbations_shape_dtype_and_determinism():
    img = np.random.default_rng(0).integers(0, 255, (80, 60, 3), dtype=np.uint8)
    for name in perturb.PERTURBATIONS:
        for sev in (1, 2, 3):
            a = perturb.apply(name, img, sev, seed=7)
            b = perturb.apply(name, img, sev, seed=7)
            assert a.shape == img.shape and a.dtype == np.uint8, name
            assert np.array_equal(a, b), f"{name} not deterministic"
        assert np.array_equal(perturb.apply(name, img, 0, seed=7), img)


def test_perturbation_strength_increases():
    img = np.full((64, 64, 3), 128, np.uint8)
    img[16:48, 16:48] = 220
    diffs = [np.abs(perturb.apply("blur", img, s, 0).astype(int) - img).mean() for s in (1, 2, 3)]
    assert diffs[0] < diffs[1] < diffs[2]


def test_dataset_perturbation_adapter_is_picklable():
    import pickle
    f = pickle.loads(pickle.dumps(perturb.DatasetPerturbation(perturb.rotation, 2)))
    img = np.zeros((32, 32, 3), np.uint8)
    assert f(img, np.random.default_rng(0), "x.png").shape == img.shape


def test_background_replacer():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "masks").mkdir()
        (d / "bg").mkdir()
        mask = np.zeros((40, 40), np.uint8)
        mask[10:30, 10:30] = 255
        cv2.imwrite(str(d / "masks" / "a.png"), mask)
        for i in range(3):
            cv2.imwrite(str(d / "bg" / f"{i}.jpg"), np.full((50, 50, 3), 60 * i + 30, np.uint8))
        rep = perturb.BackgroundReplacer(d / "masks", d / "bg")
        img = np.full((40, 40, 3), 250, np.uint8)
        out = rep(img, 2, np.random.default_rng(0), "a.jpg")
        assert out[20, 20].min() > 200            # foreground kept
        assert out[2, 2].max() < 200              # background replaced


# ---------------------------------------------------------------- smoothing
def noisy_stream(rng, n_per=60, classes=(0, 1, 2), flip_p=0.3, n_cls=5):
    probs, truth = [], []
    for c in classes:
        for _ in range(n_per):
            k = c if rng.random() > flip_p else int(rng.integers(0, n_cls))
            p = np.full(n_cls, 0.05)
            p[k] = 0.8
            probs.append(p / p.sum())
            truth.append(c)
    return np.array(probs), truth


def test_smoothing_reduces_flicker_and_keeps_accuracy():
    probs, truth = noisy_stream(np.random.default_rng(0))
    raw = smoothing.stability_metrics(smoothing.run_smoother(smoothing.build_smoother("raw"), probs), truth)
    maj = smoothing.stability_metrics(
        smoothing.run_smoother(smoothing.build_smoother("majority", window=9, threshold=0.0, min_agree=0.0),
                               probs), truth)
    ema = smoothing.stability_metrics(
        smoothing.run_smoother(smoothing.build_smoother("ema", alpha=0.2, threshold=0.0), probs), truth)
    assert maj["switches"] < raw["switches"] / 4
    assert ema["switches"] < raw["switches"] / 4
    assert maj["frame_accuracy"] > raw["frame_accuracy"]
    assert maj["latency_frames"] >= raw["latency_frames"]  # smoothing costs delay


def test_majority_threshold_withholds_uncertain():
    sm = smoothing.MajorityVote(window=4, min_agree=0.75, threshold=0.5)
    outs = [sm.update(np.eye(3)[i % 3])[0] for i in range(8)]
    assert all(o is None for o in outs[3:])  # labels keep changing -> never confident


def test_stability_metrics_counts():
    m = smoothing.stability_metrics([0, 0, 1, 1, 1, 2], [0, 0, 1, 1, 1, 1], fps=6)
    assert m["switches"] == 2 and abs(m["switches_per_min"] - 120) < 1e-6
    assert abs(m["frame_accuracy"] - 5 / 6) < 1e-9 and m["latency_frames"] == 0


# ---------------------------------------------------------------- video
def test_sample_indices():
    rng = np.random.default_rng(0)
    for n in (1, 5, 16, 32, 100):
        for train in (False, True):
            idx = video.sample_indices(n, 16, train, rng)
            assert len(idx) == 16 and idx.min() >= 0 and idx.max() <= n - 1
            assert np.all(np.diff(idx) >= 0), "temporal order must be preserved"
    assert video.uniform_indices(32, 16)[0] == 0 and video.uniform_indices(32, 16)[-1] == 31


def test_extract_frames_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        w = cv2.VideoWriter(str(d / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
        for i in range(40):
            w.write(np.full((48, 64, 3), i * 5, np.uint8))
        w.release()
        n = video.extract_frames(d / "v.mp4", d / "frames", num_frames=12, short_side=32)
        files = sorted((d / "frames").glob("*.jpg"))
        assert n == 12 and len(files) == 12
        assert cv2.imread(str(files[0])).shape[0] == 32
        first, last = cv2.imread(str(files[0])).mean(), cv2.imread(str(files[-1])).mean()
        assert last > first  # order preserved


# ---------------------------------------------------------------- metrics
def test_metrics_and_plots():
    y = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    p = np.array([0, 0, 1, 2, 2, 2, 2, 1])
    probs = np.eye(3)[p] * 0.7 + 0.1
    m = compute_metrics(y, p, ["a", "b", "c"], probs)
    assert abs(m["accuracy"] - 0.75) < 1e-9
    assert 0 < m["macro_f1"] < 1 and m["top5_accuracy"] == 1.0
    assert m["most_confused"][0]["count"] == 1
    assert len(per_class_report(y, p, ["a", "b", "c"])) == 3
    with tempfile.TemporaryDirectory() as d:
        save_confusion_matrix(y, p, ["a", "b", "c"], Path(d) / "cm.png")
        assert (Path(d) / "cm.png").stat().st_size > 1000


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
