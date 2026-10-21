# Dynamic Sign Language Recognition Using Deep Learning

Code for the project report: a static CNN baseline, lightweight transfer learning
(MobileNetV3, EfficientNet-B0), CNN + LSTM/GRU for dynamic signs, robustness and
unseen-signer evaluation, temporal smoothing and a real-time OpenCV prototype.

Every experiment in the report (Section 5.3) maps to one script:

| # | Experiment | Script | Main outputs (in `runs/<run>/`) |
|---|---|---|---|
| 1 | Custom CNN baseline | `train.py --arch custom_cnn` | `test_metrics.json`, `test_confusion_matrix.png` |
| 2 | Transfer learning | `train.py --arch mobilenet_v3_large / efficientnet_b0`, then `evaluate.py` | + `efficiency.json` (params, MB, ms, FPS) |
| 3 | Robustness to image variation | `robustness.py` | `robustness.csv`, `robustness.png` |
| 4 | User-independent test | `prepare_*.py --mode signer` + `compare_results.py` | `results/generalisation_gap.csv` |
| 5 | Dynamic CNN+LSTM (and GRU, and no-temporal ablation) | `train.py --task dynamic --rnn lstm/gru/none` | same as 1 |
| 6 | Temporal smoothing | `stability.py` | `stability.csv`, `stability_trace.png` |
| 7 | Real-time OpenCV prototype | `webcam.py` | live window, optional video + per-frame CSV log |
| – | All results in one table | `compare_results.py` | `results/results.md`, `summary.csv` |

---

## 1. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# NVIDIA GPU: install the CUDA build of torch first, from the selector at https://pytorch.org
python smoke_test.py
```

`smoke_test.py` generates a small synthetic dataset and runs **every** script end to end
(about 2–5 minutes on a CPU, no downloads). If it ends with `SMOKE TEST PASSED`, your
environment is ready. The synthetic data is only there to test the code. Its numbers mean nothing.

## 2. Data

### Static (Experiments 1–4)

`prepare_static.py` accepts `root/<class>/<img>` or `root/<signer>/<class>/<img>` (or a
`--signer-regex` that reads the signer from file names).

* **Kaggle "ASL Alphabet"** works for Experiments 1–3, with two caveats the script surfaces
  for you. It essentially has one signer, so a signer-independent split is impossible.
  Its frames are near-duplicates, which inflates random-split accuracy. Run with
  `--check-duplicates` and report the near-duplicate rate. Consider `--exclude-classes J Z`
  (motion letters) and possibly `nothing space del`.
* For **Experiment 4** you need a dataset with signer ids, e.g. the Massey University ASL
  gesture dataset (5 signers) or your own photos from several classmates. Check the licence
  and availability of any dataset before using it.

```bash
python prepare_static.py --root data/asl_alphabet/asl_alphabet_train/asl_alphabet_train --mode random --exclude-classes J Z --check-duplicates --out splits/static_random.csv
python prepare_static.py --root data/multi_signer --layout signer/class --mode signer --out splits/static_signer.csv
```

Each call writes the split CSV and a `.report.json` with the signer overlap and warnings. Quote
these in the report's evaluation-protocol section.

### Dynamic (Experiment 5)

`prepare_video.py` extracts a fixed number of uniformly spaced frames per clip (default 32)
and writes the split CSV. Recommended: **WLASL-100** (manageable size, official split) or
**AUTSL** (official split is signer-independent, which matches Experiment 4).

```bash
# WLASL-100 (videos from the official download/scraper; many source URLs are dead,
# so the script reports how many were found - put that number in the report)
python prepare_video.py --wlasl-json data/WLASL_v0.3.json --videos-root data/wlasl/videos --top-k 100 --frames-root data/frames_wlasl100 --mode predefined --out splits/wlasl100.csv

# AUTSL (verify the label-file format against your download; see from_autsl())
python prepare_video.py --autsl-labels train=data/autsl/train_labels.csv val=data/autsl/val_labels.csv test=data/autsl/test_labels.csv --autsl-classes data/autsl/SignList_ClassId_TR_EN.csv --videos-root data/autsl --frames-root data/frames_autsl --mode predefined --out splits/autsl.csv

# Your own recordings: videos/<signer>/<sign>/<clip>.mp4
python prepare_video.py --videos-root data/my_videos --layout signer/class --frames-root data/frames_mine --mode signer --out splits/mine_signer.csv
```

## 3. Running the experiments

Run each configuration with 3 seeds (`--seed 0/1/2`); `compare_results.py` reports mean ± std.

```bash
# Exp 1 - baseline from scratch
python train.py --task static --arch custom_cnn --data-root data/asl_alphabet/asl_alphabet_train/asl_alphabet_train --split-csv splits/static_random.csv --epochs 30

# Exp 2 - transfer learning (head only for 2 epochs, then fine-tune with backbone lr x0.1)
python train.py --task static --arch mobilenet_v3_large --data-root data/asl_alphabet/asl_alphabet_train/asl_alphabet_train --split-csv splits/static_random.csv --epochs 15 --freeze-epochs 2
python train.py --task static --arch efficientnet_b0 --data-root data/asl_alphabet/asl_alphabet_train/asl_alphabet_train --split-csv splits/static_random.csv --epochs 15 --freeze-epochs 2
python evaluate.py --run runs/static_mobilenet_v3_large_static_random_seed0   # adds latency/size

# Exp 3 - robustness
python robustness.py --run runs/static_mobilenet_v3_large_static_random_seed0

# Exp 4 - same model, unseen-signer split (then compare_results computes the gap)
python train.py --task static --arch mobilenet_v3_large --data-root data/multi_signer --split-csv splits/static_signer.csv --epochs 15 --freeze-epochs 2

# Exp 5 - dynamic: LSTM, GRU, and the ablation without a temporal model
python train.py --task dynamic --arch mobilenet_v3_small --rnn lstm --seq-len 16 --data-root data/frames_wlasl100 --split-csv splits/wlasl100.csv --epochs 40 --batch-size 8
python train.py --task dynamic --arch mobilenet_v3_small --rnn gru  ...same...
python train.py --task dynamic --arch mobilenet_v3_small --rnn none ...same...
#   low on GPU memory?  add --freeze-backbone (only the LSTM + head train; much faster)

# Exp 6 - smoothing: raw vs majority vote vs EMA, flicker and delay
python stability.py --run runs/<run>
python stability.py --run runs/<run> --video my_take.mp4 --labels my_take_labels.csv  # real footage

# Exp 7 - live demo
python webcam.py --run runs/static_mobilenet_v3_large_static_random_seed0
python webcam.py --run runs/<dynamic run> --stride 4 --save-video demo.mp4 --log demo.csv

# Results table
python compare_results.py --runs runs --out results
```

**Choosing `--stride` for the dynamic live demo.** The model sees `seq-len` frames per
window. With `--seq-len 16 --stride 4`, the window covers 64 camera frames, about 2.1 s at
30 FPS, which is roughly one sign. If your training clips are shorter or longer, adjust the
stride so the window matches.

## 4. Design decisions (and where they answer the report's gaps)

* **Augmentation only in training.** Evaluation, robustness tests and the webcam use the
  deterministic resize + normalise path (`slr/data.py`).
* **No horizontal flip by default.** It swaps the dominant hand and can change
  direction-dependent signs. `--hflip` turns it on as an explicit experiment.
* **Clip-consistent augmentation.** All frames of a clip get identical random augmentation,
  so augmentation never invents motion. Temporal sampling jitters speed but always keeps
  frame order.
* **Leakage checks.** Split reports list signer overlap, and `--check-duplicates` estimates
  how many test images are near-copies of training images (perceptual hash).
* **Model selection on validation macro-F1**, early stopping, label smoothing, AdamW + cosine
  schedule, seeds saved with every run.
* **Efficient live inference.** For dynamic models, the CNN runs once per sampled frame and
  its features are cached. Only the LSTM/GRU re-runs on the window (`slr/live.py`).
* **Robustness perturbations** are deterministic per sample (and identical across a clip's
  frames). The background test (`--mask-root --background-dir`) needs foreground masks,
  e.g. from a segmentation model or green-screen recordings.
* **`--rnn none` ablation.** This averages frame features, so it cannot use frame order.
  The gap between it and LSTM/GRU is your evidence that temporal modelling matters.

## 5. Project layout

```
slr/            library code
  splits.py       indexing, random / signer-independent splits, leakage checks
  data.py         datasets + transforms (static images, frame sequences)
  video.py        frame extraction, temporal sampling
  models.py       custom CNN, MobileNetV3, EfficientNet-B0, CNN+LSTM/GRU
  engine.py       training loop, prediction, latency benchmark
  evaluation.py   metrics + artefacts for a split
  metrics.py      accuracy, macro-F1, top-5, confusion matrix, per-class report
  perturb.py      robustness perturbations
  smoothing.py    majority vote, EMA, flicker / delay metrics
  live.py         frame-by-frame predictor used by webcam + stability
prepare_static.py  prepare_video.py  train.py  evaluate.py  robustness.py
stability.py  webcam.py  compare_results.py  smoke_test.py
scripts/make_synthetic_data.py   tests/test_core.py
```

## 6. Known limitations / next steps

* The static live demo uses a fixed on-screen box for the hand. A hand detector
  (e.g. MediaPipe Hands) to crop the hand automatically is the natural next step.
* A pose-based baseline (MediaPipe keypoints → GRU) would be a cheap and robust
  comparison for Experiment 5.
* Isolated recognition only. Continuous signing and translation are out of scope, as the
  report states.
