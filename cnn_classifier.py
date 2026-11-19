"""
cnn_classifier.py - Lightweight CNNs for eye-state and yawn classification + fusion.

Sec. III-B of Ersoy et al. (2026): two separate CNNs on 64x64 crops,
    Eye CNN   : open (0) vs closed (1)
    Mouth CNN : neutral / no-yawn (0) vs yawning (1)

Here the two networks are ~60k-parameter conv nets (3 -> 16 -> 32 -> 64 -> 64 channels,
global average pooling), small enough for >1000 patches/s on a laptop CPU.

Fusion (an extension of the paper, which evaluates geometry and CNN independently):
the calibrated geometric evidence (driver-specific sigmoid score around T_ear / T_mar)
and the CNN probability are blended in the probability domain.  When the geometry is
untrustworthy (head turned, face partially lost) the CNN is used alone; when no
trained weights exist the geometry is used alone.  A hysteresis comparator turns the
fused probability into a stable binary state.

No pretrained weights are shipped.  Train them with `train_cnn.py` (MRL Eye / Yawn
datasets, as in the paper) and place them under models/.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from config import CNNConfig, CropConfig

LOG = logging.getLogger(__name__)

EYE_CLASSES: Tuple[str, str] = ("open", "closed")
MOUTH_CLASSES: Tuple[str, str] = ("neutral", "yawning")


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class StateCNN(nn.Module):
    """Tiny binary classifier for 64x64 (or any >=16x16) image patches."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 width: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width),          # 64 -> 32
            _block(width, width * 2),            # 32 -> 16
            _block(width * 2, width * 4),        # 16 -> 8
            _block(width * 4, width * 4),        # 8  -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (N, C, H, W) -> logits (N, 2)
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Pre-processing & checkpoints
# --------------------------------------------------------------------------- #
def patches_to_tensor(patches: Sequence[np.ndarray]) -> torch.Tensor:
    """uint8 BGR patches (H, W, 3) -> float tensor (N, 3, H, W) in [-1, 1], RGB order."""
    arr = np.stack(patches, axis=0)[..., ::-1]                       # BGR -> RGB
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float()
    return t.div_(127.5).sub_(1.0)


def save_checkpoint(model: StateCNN, path: Union[str, Path], task: str,
                    class_names: Sequence[str], width: int, extra: Optional[Dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "meta": {"task": task, "class_names": list(class_names), "width": width,
                         **(extra or {})}}, str(path))


def load_checkpoint(path: Union[str, Path], cfg: CNNConfig) -> StateCNN:
    ckpt = torch.load(str(path), map_location=cfg.device, weights_only=True)
    width = int(ckpt.get("meta", {}).get("width", cfg.width))
    model = StateCNN(width=width, dropout=cfg.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(cfg.device)


# --------------------------------------------------------------------------- #
# Inference wrapper
# --------------------------------------------------------------------------- #
@dataclass
class CNNOutput:
    eye_closed_prob: Optional[float] = None     # mean over both eyes
    yawn_prob: Optional[float] = None
    fresh: bool = False                         # True if computed on this frame


class CNNStateClassifier:
    """Runs the eye and mouth CNNs (with frame striding) and caches the last result."""

    def __init__(self, cfg: CNNConfig, crop_cfg: Optional[CropConfig] = None) -> None:
        self.cfg = cfg
        torch.set_num_threads(max(1, cfg.num_threads))
        self.eye_model: Optional[StateCNN] = None
        self.mouth_model: Optional[StateCNN] = None
        if cfg.enabled:
            self.eye_model = self._try_load(cfg.eye_weights, "eye")
            self.mouth_model = self._try_load(cfg.mouth_weights, "mouth")
        self._frame = 0
        self._last = CNNOutput()

    def _try_load(self, path: Path, name: str) -> Optional[StateCNN]:
        if not Path(path).exists():
            LOG.warning("No %s CNN weights at %s - %s branch runs geometry-only.", name, path, name)
            return None
        try:
            model = load_checkpoint(path, self.cfg)
            LOG.info("Loaded %s CNN (%d parameters) from %s", name, count_parameters(model), path)
            return model
        except Exception as exc:
            LOG.error("Could not load %s CNN from %s: %s", name, path, exc)
            return None

    @property
    def eye_available(self) -> bool:
        return self.eye_model is not None

    @property
    def mouth_available(self) -> bool:
        return self.mouth_model is not None

    @property
    def available(self) -> bool:
        return self.eye_available or self.mouth_available

    def reset(self) -> None:
        self._frame = 0
        self._last = CNNOutput()

    @torch.inference_mode()
    def predict(self, eye_patches: Optional[Sequence[np.ndarray]],
                mouth_patch: Optional[np.ndarray]) -> CNNOutput:
        """Return class-1 probabilities; re-uses the previous result between strides."""
        self._frame += 1
        stride = max(1, self.cfg.inference_stride)
        if (self._frame - 1) % stride != 0:
            return CNNOutput(self._last.eye_closed_prob, self._last.yawn_prob, fresh=False)

        eye_p: Optional[float] = None
        yawn_p: Optional[float] = None
        if self.eye_model is not None and eye_patches:
            logits = self.eye_model(patches_to_tensor(list(eye_patches)).to(self.cfg.device))
            eye_p = float(torch.softmax(logits, dim=1)[:, 1].mean().item())
        if self.mouth_model is not None and mouth_patch is not None:
            logits = self.mouth_model(patches_to_tensor([mouth_patch]).to(self.cfg.device))
            yawn_p = float(torch.softmax(logits, dim=1)[0, 1].item())
        self._last = CNNOutput(eye_p, yawn_p, fresh=True)
        return self._last


# --------------------------------------------------------------------------- #
# Fusion of geometry and CNN
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FusionResult:
    probability: float
    source: str          # "fused" | "geometry" | "cnn" | "none"


def fuse_probabilities(geometry_prob: Optional[float], cnn_prob: Optional[float],
                       geometry_weight: float) -> FusionResult:
    """Blend calibrated-geometry evidence with CNN evidence.

    geometry_prob is None when EAR/MAR are not trustworthy (head rotated, shadows
    breaking the landmarks); cnn_prob is None when the CNN is unavailable.
    """
    if geometry_prob is not None and cnn_prob is not None:
        w = float(min(max(geometry_weight, 0.0), 1.0))
        return FusionResult(w * geometry_prob + (1.0 - w) * cnn_prob, "fused")
    if geometry_prob is not None:
        return FusionResult(float(geometry_prob), "geometry")
    if cnn_prob is not None:
        return FusionResult(float(cnn_prob), "cnn")
    return FusionResult(0.0, "none")


class HysteresisDecision:
    """Schmitt-trigger on a probability stream: removes frame-to-frame flicker."""

    def __init__(self, enter_prob: float, exit_prob: float) -> None:
        if not 0.0 <= exit_prob <= enter_prob <= 1.0:
            raise ValueError("require 0 <= exit_prob <= enter_prob <= 1")
        self.enter_prob = enter_prob
        self.exit_prob = exit_prob
        self.state = False

    def update(self, p: float) -> bool:
        if self.state:
            if p < self.exit_prob:
                self.state = False
        elif p > self.enter_prob:
            self.state = True
        return self.state

    def reset(self) -> None:
        self.state = False


def _selftest() -> None:    # pragma: no cover - quick manual check
    m = StateCNN()
    x = torch.randn(2, 3, 64, 64)
    print("params:", count_parameters(m), "out:", tuple(m(x).shape))


if __name__ == "__main__":  # pragma: no cover
    _selftest()
