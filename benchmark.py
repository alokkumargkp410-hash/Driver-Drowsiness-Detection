"""
benchmark.py - Reproducible benchmarks for the project.

1. `simulation`  Monte-Carlo study of *static global* vs *personalised* thresholds on
                 synthetic drivers whose eye / mouth geometry differs.  It validates the
                 calibration logic and quantifies WHY a global cut-off fails across
                 anatomies.  It is a simulation: absolute numbers are NOT comparable with
                 the paper's dataset results (see README).
2. `latency`     Per-component timing (CNN, calibration scoring, state machine) and the
                 resulting throughput ceiling, excluding the MediaPipe landmark stage.
3. `video`       End-to-end FPS on a real video / webcam (needs the MediaPipe model).

    python benchmark.py simulation --subjects 200
    python benchmark.py latency
    python benchmark.py video --source drive.mp4 --frames 600
"""
from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from calibration import CalibrationError, DriverProfile, compute_profile
from config import DEFAULT_CONFIG


# --------------------------------------------------------------------------- #
# 1. Simulation
# --------------------------------------------------------------------------- #
def simulate_subject(rng: np.random.Generator, n_calib: int = 150, n_test: int = 3000) -> Dict[str, np.ndarray]:
    """Synthetic subject: individual open-eye EAR, neutral MAR and yawn MAR."""
    ear_open = rng.uniform(0.20, 0.38)            # eye anatomy (narrow ... wide)
    mar_rest = rng.uniform(0.01, 0.08)            # lip geometry at rest
    mar_yawn = rng.uniform(0.45, 0.95)
    noise = 0.04 * ear_open

    def ear_open_samples(n: int) -> np.ndarray:
        return rng.normal(ear_open, noise, n)

    def ear_closed_samples(n: int) -> np.ndarray:
        return ear_open * rng.uniform(0.10, 0.50, n) + rng.normal(0, noise * 0.5, n)

    def mar_neutral(n: int) -> np.ndarray:
        return np.abs(rng.normal(mar_rest, 0.10 * mar_rest + 0.002, n))

    def mar_yawning(n: int) -> np.ndarray:
        return rng.uniform(0.7, 1.0, n) * mar_yawn

    # calibration window: awake, eyes open, ~8% blinks, mouth closed
    blink = rng.random(n_calib) < 0.08
    ear_c = ear_open_samples(n_calib)
    ear_c[blink] = ear_closed_samples(blink.sum())
    out = {"ear_calib": ear_c, "mar_calib": mar_neutral(n_calib),
           "head_calib": rng.normal(0.8, 0.01, n_calib)}

    # labelled test frames
    n_closed = n_test // 4
    out["ear_test"] = np.concatenate([ear_open_samples(n_test - n_closed), ear_closed_samples(n_closed)])
    out["ear_label"] = np.concatenate([np.zeros(n_test - n_closed), np.ones(n_closed)]).astype(int)
    n_yawn = n_test // 4
    out["mar_test"] = np.concatenate([mar_neutral(n_test - n_yawn), mar_yawning(n_yawn)])
    out["mar_label"] = np.concatenate([np.zeros(n_test - n_yawn), np.ones(n_yawn)]).astype(int)
    return out


def _acc(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:
    """(overall accuracy, recall of class 0, recall of class 1)."""
    return (float((pred == label).mean()),
            float((pred[label == 0] == 0).mean()),
            float((pred[label == 1] == 1).mean()))


def run_simulation(n_subjects: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cal = DEFAULT_CONFIG.calibration
    subjects = [simulate_subject(rng) for _ in range(n_subjects)]
    base_ear = np.array([np.median(s["ear_calib"]) for s in subjects])
    base_mar = np.array([np.median(s["mar_calib"]) for s in subjects])

    rows: Dict[str, List[Tuple[float, float, float]]] = {
        "EAR global (ratio of population mean)": [], "EAR personalised (ratio rule, paper)": [],
        "EAR personalised (median - k*MAD/0.6745)": [],
        "MAR global (ratio of population mean)": [], "MAR personalised (ratio rule, paper)": [],
        "MAR personalised (median + k*MAD/0.6745)": [],
    }
    failed = 0
    for i, s in enumerate(subjects):
        # Global thresholds are derived from the OTHER subjects only (leave-one-out),
        # so the legacy baseline is not tuned on the driver being tested.
        others = np.arange(n_subjects) != i
        g_ear = cal.paper_ear_ratio * base_ear[others].mean()
        g_mar = cal.paper_mar_ratio * base_mar[others].mean()
        try:
            p_mad = compute_profile(s["ear_calib"], s["mar_calib"], s["head_calib"], cal)
            p_ratio = compute_profile(s["ear_calib"], s["mar_calib"], s["head_calib"], replace(cal, mode="ratio"))
        except CalibrationError:
            failed += 1
            continue
        e, el = s["ear_test"], s["ear_label"]
        m, ml = s["mar_test"], s["mar_label"]
        rows["EAR global (ratio of population mean)"].append(_acc((e < g_ear).astype(int), el))
        rows["EAR personalised (ratio rule, paper)"].append(_acc((e < p_ratio.t_ear).astype(int), el))
        rows["EAR personalised (median - k*MAD/0.6745)"].append(_acc((e < p_mad.t_ear).astype(int), el))
        rows["MAR global (ratio of population mean)"].append(_acc((m > g_mar).astype(int), ml))
        rows["MAR personalised (ratio rule, paper)"].append(_acc((m > p_ratio.t_mar).astype(int), ml))
        rows["MAR personalised (median + k*MAD/0.6745)"].append(_acc((m > p_mad.t_mar).astype(int), ml))

    print(f"\nSimulation: {n_subjects} synthetic drivers, seed={seed}, calibration failures={failed}")
    print("(validates the calibration logic; NOT a reproduction of the paper's dataset results)\n")
    print(f"{'Method':<46}{'Accuracy':>10}{'Rec(neg)':>10}{'Rec(pos)':>10}{'Worst-10% acc':>15}")
    for name, vals in rows.items():
        a = np.array(vals)
        worst = np.percentile(a[:, 0], 10)
        print(f"{name:<46}{a[:, 0].mean():>10.1%}{a[:, 1].mean():>10.1%}{a[:, 2].mean():>10.1%}{worst:>15.1%}")
    print("\nRec(neg): open eye / neutral mouth recognised;  Rec(pos): closed eye / yawn recognised.")


# --------------------------------------------------------------------------- #
# 2. Latency
# --------------------------------------------------------------------------- #
def run_latency(iterations: int) -> None:
    import torch
    from cnn_classifier import CNNStateClassifier, StateCNN, count_parameters, save_checkpoint
    from state_machine import DrowsinessStateMachine, FrameObservation

    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        save_checkpoint(StateCNN(), d / "e.pt", "eye", ("open", "closed"), DEFAULT_CONFIG.cnn.width)
        save_checkpoint(StateCNN(), d / "m.pt", "mouth", ("neutral", "yawning"), DEFAULT_CONFIG.cnn.width)
        every = replace(DEFAULT_CONFIG.cnn, eye_weights=d / "e.pt", mouth_weights=d / "m.pt", inference_stride=1)
        stride = replace(every, inference_stride=DEFAULT_CONFIG.cnn.inference_stride)
        print(f"CNN parameters per network: {count_parameters(StateCNN(width=DEFAULT_CONFIG.cnn.width)):,}")
        for label, cfg in (("every frame", every), (f"stride {stride.inference_stride}", stride)):
            clf = CNNStateClassifier(cfg)
            for _ in range(20):
                clf.predict([patch, patch], patch)
            t0 = time.perf_counter()
            for _ in range(iterations):
                clf.predict([patch, patch], patch)
            ms = (time.perf_counter() - t0) / iterations * 1000
            print(f"CNN eye(2 patches)+mouth, {label:<12}: {ms:6.3f} ms/frame  ({torch.get_num_threads()} threads)")

    prof = compute_profile(rng.normal(0.3, 0.012, 150), np.abs(rng.normal(0.03, 0.004, 150)),
                           rng.normal(0.8, 0.01, 150), DEFAULT_CONFIG.calibration)
    t0 = time.perf_counter()
    for i in range(iterations * 10):
        prof.eye_closed_score(0.28)
        prof.mouth_yawn_score(0.05)
    print(f"Calibrated geometric scores        : {(time.perf_counter() - t0) / (iterations * 10) * 1000:6.4f} ms/frame")

    sm = DrowsinessStateMachine(DEFAULT_CONFIG.state)
    t0 = time.perf_counter()
    for i in range(iterations * 10):
        sm.update(FrameObservation(i / 30.0, True, (i // 7) % 5 == 0, False, False))
    print(f"State machine (60 s window)        : {(time.perf_counter() - t0) / (iterations * 10) * 1000:6.4f} ms/frame")
    print("\nThe MediaPipe landmark stage is the dominant cost; measure it end-to-end with:\n"
          "  python benchmark.py video --source <file or 0>")


# --------------------------------------------------------------------------- #
# 3. End-to-end video
# --------------------------------------------------------------------------- #
def run_video(source: str, frames: int, use_cnn: bool) -> None:
    import cv2
    from main import DrowsinessPipeline

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    pipe = DrowsinessPipeline(DEFAULT_CONFIG, use_cnn=use_cnn)
    lat: List[float] = []
    n = 0
    t_start = time.perf_counter()
    while n < frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (DEFAULT_CONFIG.video.frame_width, DEFAULT_CONFIG.video.frame_height))
        res = pipe.process(frame, n / 30.0)
        lat.append(res.latency_ms)
        n += 1
    total = time.perf_counter() - t_start
    pipe.close()
    cap.release()
    if not lat:
        raise SystemExit("no frames processed")
    a = np.array(lat)
    print(f"{n} frames | mean {a.mean():.1f} ms | p95 {np.percentile(a, 95):.1f} ms | "
          f"throughput {n / total:.1f} FPS | phase at end: {pipe.phase}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulation")
    s.add_argument("--subjects", type=int, default=200)
    s.add_argument("--seed", type=int, default=7)
    l = sub.add_parser("latency")
    l.add_argument("--iterations", type=int, default=300)
    v = sub.add_parser("video")
    v.add_argument("--source", default="0")
    v.add_argument("--frames", type=int, default=600)
    v.add_argument("--no-cnn", action="store_true")
    a = ap.parse_args()
    if a.cmd == "simulation":
        run_simulation(a.subjects, a.seed)
    elif a.cmd == "latency":
        run_latency(a.iterations)
    else:
        run_video(a.source, a.frames, not a.no_cnn)


if __name__ == "__main__":
    main()
