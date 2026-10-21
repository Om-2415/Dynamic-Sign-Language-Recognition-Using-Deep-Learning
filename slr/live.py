"""Frame-by-frame inference used by the webcam app and the stability experiment."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torchvision import tv_tensors

from .data import build_transforms


class LivePredictor:
    """Feed RGB frames one at a time; get class probabilities back.

    static:  every frame is classified on its own.
    dynamic: every `stride`-th frame goes through the CNN once and its feature vector is
             cached; the LSTM/GRU then runs over the last `seq_len` cached features.
             The window therefore spans seq_len * stride camera frames - choose stride so
             this matches the duration of a sign in the training clips (about 1.5-2.5 s).
    """

    def __init__(self, model, ckpt: dict, device, stride: int = 1):
        ta = ckpt["train_args"]
        self.model, self.device = model.eval(), device
        self.task, self.classes = ta["task"], ckpt["classes"]
        self.seq_len = ta.get("seq_len", 16)
        self.stride = max(1, stride)
        self.tf = build_transforms(ta["img_size"], train=False)
        self.feats: deque[torch.Tensor] = deque(maxlen=self.seq_len)
        self.count = 0
        self.last: np.ndarray | None = None

    def reset(self):
        self.feats.clear()
        self.count = 0
        self.last = None

    def preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        x = self.tf(tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)))
        return x.as_subclass(torch.Tensor).to(self.device)

    @torch.no_grad()
    def step(self, rgb: np.ndarray) -> np.ndarray | None:
        """Returns probabilities, or None while the dynamic buffer is still filling."""
        if self.task == "static":
            logits = self.model(self.preprocess(rgb)[None])
            self.last = torch.softmax(logits, 1)[0].cpu().numpy()
            return self.last
        if self.count % self.stride == 0:
            x = self.preprocess(rgb)[None]
            self.feats.append(self.model.backbone(x)[0])
            if len(self.feats) == self.seq_len:
                f = torch.stack(list(self.feats))[None]  # 1,T,D
                self.last = torch.softmax(self.model.classify_features(f), 1)[0].cpu().numpy()
        self.count += 1
        return self.last
