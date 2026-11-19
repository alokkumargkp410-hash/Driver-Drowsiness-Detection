"""
train_cnn.py - Train the eye-state or mouth-state CNN (paper Sec. IV).

Dataset layout (folder names are matched case-insensitively):

    eye task    : <data_dir>/{open,closed}/*.png|jpg        (e.g. MRL Eye Dataset, re-sorted)
    mouth task  : <data_dir>/{no_yawn,yawn}/*.png|jpg       (e.g. Kaggle Yawn Dataset)

Accepted aliases: open / open_eyes / open_eye ;  closed / close / closed_eyes / closed_eye ;
                  no_yawn / noyawn / neutral / closed_mouth ;  yawn / yawning / open_mouth.

Protocol follows the paper: 70% train / 15% validation / 15% test, 64x64 inputs,
CLAHE enhancement, augmentation by horizontal flip, random brightness and small
rotations.  The best-validation checkpoint is written to models/eye_cnn.pt or
models/mouth_cnn.pt together with the test accuracy and confusion matrix.

    python train_cnn.py --task eye   --data-dir data/eyes  --epochs 15
    python train_cnn.py --task mouth --data-dir data/yawn  --epochs 20
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cnn_classifier import (EYE_CLASSES, MOUTH_CLASSES, StateCNN, count_parameters,
                            patches_to_tensor, save_checkpoint)
from config import DEFAULT_CONFIG, MODELS_DIR

LOG = logging.getLogger("train_cnn")

ALIASES = {
    "eye": {0: ("open", "open_eyes", "open_eye", "opened"),
            1: ("closed", "close", "closed_eyes", "closed_eye")},
    "mouth": {0: ("no_yawn", "noyawn", "neutral", "closed_mouth", "no-yawn"),
              1: ("yawn", "yawning", "open_mouth")},
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def find_samples(data_dir: Path, task: str) -> List[Tuple[Path, int]]:
    folders = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
    samples: List[Tuple[Path, int]] = []
    for label, names in ALIASES[task].items():
        match = next((folders[n] for n in names if n in folders), None)
        if match is None:
            raise FileNotFoundError(f"No folder for class {label} in {data_dir} (expected one of {names})")
        files = [f for f in match.rglob("*") if f.suffix.lower() in IMG_EXT]
        LOG.info("class %d <- %s : %d images", label, match.name, len(files))
        samples += [(f, label) for f in files]
    if not samples:
        raise FileNotFoundError("no images found")
    return samples


def split_samples(samples: Sequence[Tuple[Path, int]], seed: int):
    """Stratified 70/15/15 split."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in sorted({l for _, l in samples}):
        items = [s for s in samples if s[1] == label]
        rng.shuffle(items)
        n = len(items)
        a, b = int(0.70 * n), int(0.85 * n)
        train += items[:a]
        val += items[a:b]
        test += items[b:]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


class PatchDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], size: Tuple[int, int], augment: bool,
                 clahe_clip: float, clahe_tile: Tuple[int, int], use_clahe: bool) -> None:
        self.samples = list(samples)
        self.size = size
        self.augment = augment
        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)     # grayscale files are expanded to BGR
        if img is None:
            raise OSError(f"unreadable image {path}")
        return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = self._load(path)
        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            angle = random.uniform(-10.0, 10.0)
            m = cv2.getRotationMatrix2D((self.size[0] / 2, self.size[1] / 2), angle, 1.0)
            img = cv2.warpAffine(img, m, self.size, borderMode=cv2.BORDER_REPLICATE)
            img = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        if self.use_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return patches_to_tensor([img])[0], label


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[int(t), int(p)] += 1
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    return acc, cm


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = DEFAULT_CONFIG
    size = cfg.crops.eye_crop_size if args.task == "eye" else cfg.crops.mouth_crop_size
    classes = EYE_CLASSES if args.task == "eye" else MOUTH_CLASSES

    samples = find_samples(args.data_dir, args.task)
    tr, va, te = split_samples(samples, args.seed)
    LOG.info("split: train=%d val=%d test=%d", len(tr), len(va), len(te))

    mk = lambda s, aug: PatchDataset(s, size, aug, cfg.crops.clahe_clip_limit,
                                     cfg.crops.clahe_tile_grid, cfg.crops.use_clahe)
    loaders = (
        DataLoader(mk(tr, True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers),
        DataLoader(mk(va, False), batch_size=args.batch_size, num_workers=args.workers),
        DataLoader(mk(te, False), batch_size=args.batch_size, num_workers=args.workers),
    )
    device = args.device
    model = StateCNN(width=cfg.cnn.width, dropout=cfg.cnn.dropout).to(device)
    LOG.info("model parameters: %d", count_parameters(model))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out_path = args.output or (MODELS_DIR / f"{args.task}_cnn.pt")
    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, running = time.time(), 0, 0.0
        for x, y in loaders[0]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            total += x.size(0)
        sched.step()
        val_acc, _ = evaluate(model, loaders[1], device)
        LOG.info("epoch %2d/%d loss=%.4f val_acc=%.4f (%.1fs)", epoch, args.epochs,
                 running / max(total, 1), val_acc, time.time() - t0)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_acc, cm = evaluate(model, loaders[2], device)
    LOG.info("TEST accuracy = %.4f", test_acc)
    LOG.info("confusion matrix (rows=true %s, cols=pred):\n%s", list(classes), cm)
    save_checkpoint(model.cpu(), out_path, args.task, classes, cfg.cnn.width,
                    extra={"val_acc": best_acc, "test_acc": test_acc, "input_size": list(size)})
    LOG.info("saved %s", out_path)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the eye / mouth state CNN")
    p.add_argument("--task", choices=("eye", "mouth"), required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(train(parse_args()))
