"""
main.py - Real-time Driver Drowsiness Detection pipeline (webcam or video file).

    python main.py                         # default webcam, 5 s calibration, then tracking
    python main.py --source drive.mp4      # video file
    python main.py --profile profiles/alice.json          # load profile if present, else calibrate+save
    python main.py --recalibrate --profile profiles/alice.json
    python main.py --no-cnn                # geometry-only (paper's personalised-threshold branch)
    python main.py --source drive.mp4 --no-display --output out.mp4 --log-csv log.csv

Keys:  q / ESC quit | r recalibrate | s save current profile

Pipeline per frame
    frame -> landmarks -> EAR / MAR / head ratio + eye & mouth crops
          -> [CALIBRATING] Calibrator collects N valid awake frames -> DriverProfile
          -> [TRACKING]    smoothing -> calibrated geometric score (+ CNN probability)
                           -> fused probability -> hysteresis -> binary states
                           -> 60 s sliding-window state machine -> alert level 0/1/2
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from calibration import CalibrationStatus, Calibrator, DriverProfile
from cnn_classifier import (CNNStateClassifier, HysteresisDecision, fuse_probabilities)
from config import DEFAULT_CONFIG, PROFILES_DIR, AppConfig
from landmark_extractor import FrameFeatures, LandmarkExtractor
from state_machine import (AlertStatus, DrowsinessStateMachine, FrameObservation,
                           RollingSmoother, WindowMetrics)

LOG = logging.getLogger("drowsiness")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    phase: str = "CALIBRATING"                 # "CALIBRATING" | "TRACKING"
    features: FrameFeatures = field(default_factory=FrameFeatures)
    calibration: Optional[CalibrationStatus] = None
    ear_smooth: float = float("nan")
    mar_smooth: float = float("nan")
    eye_closed_prob: float = 0.0
    yawn_prob: float = 0.0
    eye_source: str = "none"                   # fused | geometry | cnn | none
    mouth_source: str = "none"
    eye_closed: bool = False
    yawning: bool = False
    head_off_axis: bool = False
    alert: Optional[AlertStatus] = None
    cnn_active: bool = False
    latency_ms: float = 0.0


class DrowsinessPipeline:
    """Glue: landmark extraction -> calibration -> fusion -> state machine."""

    def __init__(self, cfg: AppConfig = DEFAULT_CONFIG, use_cnn: bool = True,
                 profile: Optional[DriverProfile] = None,
                 extractor: Optional[LandmarkExtractor] = None) -> None:
        self.cfg = cfg
        self.extractor = extractor if extractor is not None else LandmarkExtractor(cfg.landmarks, cfg.crops)
        self.calibrator = Calibrator(cfg.calibration)
        self.cnn: Optional[CNNStateClassifier] = None
        if use_cnn and cfg.cnn.enabled:
            self.cnn = CNNStateClassifier(cfg.cnn, cfg.crops)
            if not self.cnn.available:
                LOG.warning("CNN weights not found - running with calibrated geometry only. "
                            "Train them with train_cnn.py.")
        self.state_machine = DrowsinessStateMachine(cfg.state)
        self._eye_smoother = RollingSmoother(cfg.state.eye_smoothing_frames, use_median=True)
        self._mar_smoother = RollingSmoother(cfg.state.mar_smoothing_frames, use_median=False)
        self._eye_dec = HysteresisDecision(cfg.cnn.decision_enter_prob, cfg.cnn.decision_exit_prob)
        self._mouth_dec = HysteresisDecision(cfg.cnn.decision_enter_prob, cfg.cnn.decision_exit_prob)
        self.profile: Optional[DriverProfile] = None
        if profile is not None:
            self._set_profile(profile)

    # ------------------------------------------------------------- control -- #
    def _set_profile(self, profile: DriverProfile) -> None:
        self.profile = profile
        self._clear_tracking_state()

    def _clear_tracking_state(self) -> None:
        self.state_machine.reset()
        self._eye_smoother.clear()
        self._mar_smoother.clear()
        self._eye_dec.reset()
        self._mouth_dec.reset()
        if self.cnn is not None:
            self.cnn.reset()

    def start_calibration(self) -> None:
        self.profile = None
        self.calibrator.reset()
        self._clear_tracking_state()

    @property
    def phase(self) -> str:
        return "TRACKING" if self.profile is not None else "CALIBRATING"

    def close(self) -> None:
        self.extractor.close()

    # ---------------------------------------------------------------- run --- #
    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        t0 = time.perf_counter()
        feats = self.extractor.process(frame_bgr, t, want_patches=self.profile is not None
                                       and self.cnn is not None and self.cnn.available)
        res = self.process_features(feats, t)
        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def process_features(self, feats: FrameFeatures, t: float) -> FrameResult:
        """Everything after landmark extraction (also used by tests and benchmarks)."""
        if self.profile is None:
            return self._calibrating(feats)
        return self._tracking(feats, t)

    # ------------------------------------------------------- calibration ---- #
    def _calibrating(self, feats: FrameFeatures) -> FrameResult:
        valid = feats.face_found and feats.geometry_valid
        status = self.calibrator.update(feats.ear, feats.mar, feats.head_ratio, valid)
        if status.complete and status.profile is not None:
            self._set_profile(status.profile)
            LOG.info("Calibration complete: %s", status.profile.summary())
        return FrameResult(phase=self.phase, features=feats, calibration=status,
                           ear_smooth=feats.ear, mar_smooth=feats.mar)

    # ---------------------------------------------------------- tracking ---- #
    def _tracking(self, feats: FrameFeatures, t: float) -> FrameResult:
        prof = self.profile
        assert prof is not None
        cnn_cfg = self.cfg.cnn
        res = FrameResult(phase="TRACKING", features=feats)

        if not feats.face_found:
            res.alert = self.state_machine.update(FrameObservation(t, False, False, False, False))
            return res

        # --- geometry (only trusted when the face is frontal enough) ---------- #
        geo_eye: Optional[float] = None
        geo_mouth: Optional[float] = None
        if feats.geometry_valid:
            res.ear_smooth = self._eye_smoother.push(feats.ear)
            res.mar_smooth = self._mar_smoother.push(feats.mar)
            geo_eye = prof.eye_closed_score(res.ear_smooth, cnn_cfg.geometry_sigmoid_sharpness)
            geo_mouth = prof.mouth_yawn_score(res.mar_smooth, cnn_cfg.geometry_sigmoid_sharpness)
        else:
            res.ear_smooth = self._eye_smoother.value
            res.mar_smooth = self._mar_smoother.value

        # --- CNN --------------------------------------------------------------- #
        cnn_eye: Optional[float] = None
        cnn_mouth: Optional[float] = None
        if self.cnn is not None and self.cnn.available:
            out = self.cnn.predict(feats.eye_patches, feats.mouth_patch)
            cnn_eye, cnn_mouth = out.eye_closed_prob, out.yawn_prob
            res.cnn_active = True

        # --- fusion + hysteresis ---------------------------------------------- #
        fe = fuse_probabilities(geo_eye, cnn_eye, cnn_cfg.geometry_weight_eye)
        fm = fuse_probabilities(geo_mouth, cnn_mouth, cnn_cfg.geometry_weight_mouth)
        res.eye_closed_prob, res.eye_source = fe.probability, fe.source
        res.yawn_prob, res.mouth_source = fm.probability, fm.source
        if fe.source != "none":
            res.eye_closed = self._eye_dec.update(fe.probability)
        if fm.source != "none":
            res.yawning = self._mouth_dec.update(fm.probability)
        res.head_off_axis = bool(np.isfinite(feats.head_ratio) and feats.geometry_valid
                                 and prof.is_head_off_axis(feats.head_ratio))

        res.alert = self.state_machine.update(
            FrameObservation(t, True, res.eye_closed, res.yawning, res.head_off_axis))
        return res


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
GREEN, ORANGE, RED = (80, 200, 80), (0, 165, 255), (50, 50, 230)
WHITE, GREY, CYAN, YELLOW = (240, 240, 240), (150, 150, 150), (230, 200, 60), (0, 220, 255)


def _text(img: np.ndarray, s: str, org: Tuple[int, int], color=WHITE, scale: float = 0.5, thick: int = 1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float) -> None:
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_graph(img: np.ndarray, rect: Tuple[int, int, int, int], values: Deque[float],
               threshold: Optional[float], baseline: Optional[float], label: str,
               color: Tuple[int, int, int], alpha: float) -> None:
    x, y, w, h = rect
    _panel(img, x, y, x + w, y + h, alpha)
    cv2.rectangle(img, (x, y), (x + w, y + h), GREY, 1)
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    refs = [v for v in (threshold, baseline) if v is not None]
    pool = np.concatenate([arr, np.array(refs)]) if arr.size or refs else np.array([0.0, 1.0])
    lo, hi = float(pool.min()), float(pool.max())
    pad = max((hi - lo) * 0.15, 1e-3)
    lo, hi = lo - pad, hi + pad

    def ypix(v: float) -> int:
        return int(y + h - 1 - (v - lo) / (hi - lo) * (h - 2))

    if baseline is not None:
        cv2.line(img, (x, ypix(baseline)), (x + w, ypix(baseline)), GREY, 1, cv2.LINE_AA)
    if threshold is not None:
        cv2.line(img, (x, ypix(threshold)), (x + w, ypix(threshold)), YELLOW, 1, cv2.LINE_AA)
    vals = list(values)
    if len(vals) >= 2:
        step = w / max(len(values) - 1, 1)
        pts = [(int(x + i * step), ypix(v)) for i, v in enumerate(vals) if np.isfinite(v)]
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    _text(img, label, (x + 4, y + 14), WHITE, 0.42)


def draw_hud(frame: np.ndarray, res: FrameResult, fps: float, profile: Optional[DriverProfile],
             ear_hist: Deque[float], mar_hist: Deque[float], cfg: AppConfig) -> np.ndarray:
    hud = cfg.hud
    out = frame
    h, w = out.shape[:2]
    fs = hud.font_scale

    # landmarks overlay (eyes + mouth) for visual feedback
    feats = res.features
    if feats.face_found and feats.landmarks_px is not None:
        lm = cfg.landmarks
        for idx in lm.right_eye + lm.left_eye + lm.mouth_outer:
            px = feats.landmarks_px[idx]
            cv2.circle(out, (int(px[0]), int(px[1])), 1, CYAN, -1, cv2.LINE_AA)
        if feats.face_bbox is not None:
            x0, y0, x1, y1 = feats.face_bbox
            cv2.rectangle(out, (x0, y0), (x1, y1), GREEN if res.phase == "TRACKING" else CYAN, 1)

    # info panel
    _panel(out, 6, 6, 270, 150, hud.panel_alpha)
    _text(out, f"FPS: {fps:5.1f}   ({res.latency_ms:4.1f} ms)", (14, 26), WHITE, fs)
    state_color = GREEN if res.phase == "TRACKING" else YELLOW
    _text(out, f"PROFILE: {res.phase}", (14, 46), state_color, fs, 1)
    if profile is not None:
        _text(out, f"T_ear {profile.t_ear:.3f}  (base {profile.ear_median:.3f})", (14, 66), WHITE, fs)
        _text(out, f"T_mar {profile.t_mar:.3f}  (base {profile.mar_median:.3f})", (14, 86), WHITE, fs)
        eye_c = RED if res.eye_closed else GREEN
        mouth_c = RED if res.yawning else GREEN
        _text(out, f"EAR {res.ear_smooth:.3f} {'CLOSED' if res.eye_closed else 'open'}"
                   f" p={res.eye_closed_prob:.2f} [{res.eye_source}]", (14, 106), eye_c, fs)
        _text(out, f"MAR {res.mar_smooth:.3f} {'YAWN' if res.yawning else 'neutral'}"
                   f" p={res.yawn_prob:.2f} [{res.mouth_source}]", (14, 126), mouth_c, fs)
        _text(out, f"CNN: {'on' if res.cnn_active else 'off (geometry only)'}", (14, 144), GREY, fs)
    elif res.calibration is not None:
        cs = res.calibration
        bar_w = 240
        cv2.rectangle(out, (14, 62), (14 + bar_w, 76), GREY, 1)
        cv2.rectangle(out, (14, 62), (14 + int(bar_w * cs.progress), 76), YELLOW, -1)
        _text(out, f"{cs.collected}/{cs.target} frames", (14, 96), WHITE, fs)
        _text(out, "Thresholds are computed from YOUR baseline", (14, 116), WHITE, 0.42)

    # window metrics (right)
    if res.alert is not None:
        m = res.alert.metrics
        _panel(out, w - 214, 6, w - 6, 118, hud.panel_alpha)
        _text(out, f"Window {m.window_span_s:4.0f}s / {cfg.state.window_seconds:.0f}s", (w - 206, 26), WHITE, fs)
        _text(out, f"Blinks  N_b = {m.blink_count}", (w - 206, 46), WHITE, fs)
        _text(out, f"Micro-sleeps N_s = {m.microsleep_count}", (w - 206, 66), WHITE, fs)
        _text(out, f"Yawns   N_y = {m.yawn_count}", (w - 206, 86), WHITE, fs)
        _text(out, f"PERCLOS = {m.perclos:5.1%}", (w - 206, 106), WHITE, fs)

    # live graphs (bottom-left)
    gw, gh = hud.graph_width, hud.graph_height
    gy = h - 2 * gh - 18
    draw_graph(out, (6, gy, gw, gh), ear_hist, profile.t_ear if profile else None,
               profile.ear_median if profile else None, "EAR (yellow = T_ear)", GREEN, hud.panel_alpha)
    draw_graph(out, (6, gy + gh + 6, gw, gh), mar_hist, profile.t_mar if profile else None,
               profile.mar_median if profile else None, "MAR (yellow = T_mar)", ORANGE, hud.panel_alpha)

    # banner
    if res.alert is not None and res.alert.level > 0:
        lvl = res.alert.level
        color = RED if lvl == 2 else ORANGE
        if lvl == 2 and int(time.time() * 4) % 2 == 0:
            color = (30, 30, 160)           # flash
        cv2.rectangle(out, (0, h // 2 - 38), (w, h // 2 + 38), color, -1)
        title = "LEVEL 2 - CRITICAL DROWSINESS ALERT" if lvl == 2 else "LEVEL 1 - WARNING: SIGNS OF FATIGUE"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(out, title, ((w - tw) // 2, h // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        reasons = " | ".join(res.alert.reasons[:3])
        (rw, _), _ = cv2.getTextSize(reasons, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(out, reasons, (max((w - rw) // 2, 4), h // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    elif res.phase == "TRACKING" and not feats.face_found:
        _text(out, "FACE LOST - re-acquiring...", (w // 2 - 120, h // 2), YELLOW, 0.7, 2)
    elif res.phase == "CALIBRATING" and res.calibration is not None:
        msg = res.calibration.message
        (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        _text(out, msg, (max((w - mw) // 2, 4), h - 14), YELLOW, 0.6, 1)
    return out


# --------------------------------------------------------------------------- #
# Application loop
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Personalised driver drowsiness detection (Ersoy et al., 2026)")
    p.add_argument("--source", default="0", help="camera index or path to a video file (default: 0)")
    p.add_argument("--profile", type=Path, default=None,
                   help="JSON driver profile: loaded if it exists, saved after calibration")
    p.add_argument("--recalibrate", action="store_true", help="ignore an existing --profile and calibrate again")
    p.add_argument("--calib-mode", choices=("mad", "ratio"), default=None,
                   help="override threshold rule: mad (default) or the paper's 75%%/140%% ratio rule")
    p.add_argument("--no-cnn", action="store_true", help="disable the CNN branch")
    p.add_argument("--no-display", action="store_true", help="run headless")
    p.add_argument("--output", type=Path, default=None, help="write the annotated video here")
    p.add_argument("--log-csv", type=Path, default=None, help="write per-frame metrics to CSV")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = unlimited)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def open_source(source: str, cfg: AppConfig) -> Tuple[cv2.VideoCapture, bool]:
    is_cam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.video.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.video.frame_height)
        cap.set(cv2.CAP_PROP_FPS, cfg.video.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.video.capture_buffer_size)
    return cap, is_cam


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = DEFAULT_CONFIG
    if args.calib_mode:
        from dataclasses import replace
        cfg = replace(cfg, calibration=replace(cfg.calibration, mode=args.calib_mode))

    try:
        cap, is_cam = open_source(args.source, cfg)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 2

    file_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not is_cam and file_fps <= 1.0:
        file_fps = cfg.video.target_fps

    profile: Optional[DriverProfile] = None
    if args.profile and args.profile.exists() and not args.recalibrate:
        try:
            profile = DriverProfile.load(args.profile)
            LOG.info("Loaded driver profile %s: %s", args.profile, profile.summary())
        except Exception as exc:
            LOG.warning("Could not read profile %s (%s) - calibrating instead", args.profile, exc)

    try:
        pipeline = DrowsinessPipeline(cfg, use_cnn=not args.no_cnn, profile=profile)
    except Exception as exc:
        LOG.error("Could not start landmark backend: %s", exc)
        cap.release()
        return 3

    display = not args.no_display
    writer: Optional[cv2.VideoWriter] = None
    csv_file = None
    csv_writer = None
    if args.log_csv:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t", "phase", "face", "ear", "mar", "ear_s", "mar_s", "p_eye", "p_yawn",
                             "eye_closed", "yawning", "head_off", "level", "blinks", "microsleeps",
                             "yawns", "perclos"])

    ear_hist: Deque[float] = deque(maxlen=cfg.hud.graph_history_frames)
    mar_hist: Deque[float] = deque(maxlen=cfg.hud.graph_history_frames)
    fps = cfg.video.target_fps
    last_wall = time.perf_counter()
    start_wall = last_wall
    frame_idx = 0
    read_failures = 0
    last_level = 0

    LOG.info("Starting. Phase: %s. Press q to quit, r to recalibrate, s to save the profile.", pipeline.phase)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_cam:
                    read_failures += 1
                    if read_failures >= cfg.video.max_consecutive_read_failures:
                        LOG.error("Camera stopped delivering frames.")
                        break
                    time.sleep(0.01)
                    continue
                break                       # end of file
            read_failures = 0

            if frame.shape[1] != cfg.video.frame_width or frame.shape[0] != cfg.video.frame_height:
                frame = cv2.resize(frame, (cfg.video.frame_width, cfg.video.frame_height),
                                   interpolation=cv2.INTER_AREA)
            if is_cam and cfg.video.mirror_webcam:
                frame = cv2.flip(frame, 1)

            t = (time.perf_counter() - start_wall) if is_cam else frame_idx / file_fps
            was_calibrating = pipeline.phase == "CALIBRATING"
            res = pipeline.process(frame, t)
            if was_calibrating and pipeline.phase == "TRACKING" and args.profile:
                saved = pipeline.profile.save(args.profile)    # type: ignore[union-attr]
                LOG.info("Profile saved to %s", saved)

            now = time.perf_counter()
            inst = 1.0 / max(now - last_wall, 1e-6)
            last_wall = now
            fps += cfg.video.fps_ema_alpha * (inst - fps)

            ear_hist.append(res.ear_smooth if res.features.face_found else float("nan"))
            mar_hist.append(res.mar_smooth if res.features.face_found else float("nan"))

            if res.alert is not None and res.alert.level != last_level:
                LOG.info("t=%.1fs alert level %d -> %d (%s) %s", t, last_level, res.alert.level,
                         res.alert.label, "; ".join(res.alert.reasons))
                last_level = res.alert.level

            if csv_writer is not None:
                m = res.alert.metrics if res.alert else WindowMetrics()
                f = res.features
                csv_writer.writerow([f"{t:.3f}", res.phase, int(f.face_found), f"{f.ear:.4f}", f"{f.mar:.4f}",
                                     f"{res.ear_smooth:.4f}", f"{res.mar_smooth:.4f}",
                                     f"{res.eye_closed_prob:.3f}", f"{res.yawn_prob:.3f}",
                                     int(res.eye_closed), int(res.yawning), int(res.head_off_axis),
                                     res.alert.level if res.alert else 0, m.blink_count,
                                     m.microsleep_count, m.yawn_count, f"{m.perclos:.4f}"])

            if display or writer is not None or args.output:
                vis = draw_hud(frame, res, fps, pipeline.profile, ear_hist, mar_hist, cfg)
                if args.output and writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             file_fps if not is_cam else cfg.video.target_fps,
                                             (vis.shape[1], vis.shape[0]))
                if writer is not None:
                    writer.write(vis)
                if display:
                    try:
                        cv2.imshow(cfg.hud.window_name, vis)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        LOG.warning("No display available - continuing headless.")
                        display = False
                        key = 255
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        LOG.info("Recalibrating...")
                        pipeline.start_calibration()
                        ear_hist.clear()
                        mar_hist.clear()
                    if key == ord("s") and pipeline.profile is not None:
                        path = args.profile or (PROFILES_DIR / "driver_profile.json")
                        LOG.info("Profile saved to %s", pipeline.profile.save(path))

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    except KeyboardInterrupt:
        LOG.info("Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        pipeline.close()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_wall
    LOG.info("Processed %d frames in %.1fs (%.1f FPS average).", frame_idx, elapsed,
             frame_idx / max(elapsed, 1e-6))
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
