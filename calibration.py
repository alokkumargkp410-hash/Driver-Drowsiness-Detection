"""
calibration.py - Person-specific baseline calibration (core idea of Ersoy et al., 2026).

Before monitoring starts, the driver sits in a neutral posture (eyes open, mouth
closed, looking at the camera) for ~5 s.  From the raw EAR / MAR / head-pose
vectors collected in that window we derive *driver-specific* decision thresholds:

    sigma_ear = MAD(EAR) / 0.6745
    T_ear     = median(EAR) - k_ear * sigma_ear           (eye-closure threshold)
    T_mar     = median(MAR) + k_mar * sigma_mar           (yawn threshold)
    T_head    = k_head * sigma_head                       (allowed |deviation| of the head ratio)

where MAD(x) = median(|x - median(x)|).  Median and MAD are used instead of mean and
standard deviation because a blink or a swallow during the calibration window would
otherwise bias the baseline.

Safeguards (all expressed relative to the driver's own statistics, none absolute):
  * sigma has a floor proportional to the median (a perfectly still face gives
    MAD ~ 0 and would otherwise put the threshold on the baseline itself),
  * blink / talking outliers are rejected with a robust z-score before final stats,
  * T_ear is clamped to a band around the driver's median,
  * mode = "ratio" reproduces the paper's original 75% / 140% rule for ablations.

Note: the MAD formulation is a robust generalisation requested for this project; the
paper itself reports a ratio rule (75% of baseline EAR, 140% of baseline MAR).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from config import CalibrationConfig

ArrayLike = Union[np.ndarray, List[float]]


class CalibrationError(RuntimeError):
    """Raised when the collected calibration window is not usable."""


# --------------------------------------------------------------------------- #
# Robust statistics
# --------------------------------------------------------------------------- #
def median_absolute_deviation(x: ArrayLike) -> float:
    """MAD(x) = median(|x - median(x)|)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("MAD of an empty sample is undefined")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def robust_sigma(x: ArrayLike, mad_to_sigma: float = 0.6745) -> float:
    """Robust standard-deviation estimate: MAD / 0.6745."""
    return median_absolute_deviation(x) / mad_to_sigma


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# --------------------------------------------------------------------------- #
# Driver profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DriverProfile:
    """Personalised baseline + thresholds for a single driver."""

    ear_median: float
    ear_mad: float
    ear_sigma: float
    t_ear: float
    mar_median: float
    mar_mad: float
    mar_sigma: float
    t_mar: float
    head_median: float
    head_sigma: float
    t_head_dev: float
    n_samples: int
    mode: str
    created_at: float

    # ---- scores in the probability domain (used for CNN fusion) ------------ #
    def eye_closed_score(self, ear: float, sharpness: float = 4.0) -> float:
        """Soft geometric evidence that the eye is closed, in [0, 1].

        0.5 exactly at T_ear; ~1/(1+e^sharpness) at the driver's open-eye median.
        The slope is derived from the driver's own (median - T_ear) margin, so the
        score has no absolute scale.
        """
        margin = max(self.ear_median - self.t_ear, 1e-6)
        return _sigmoid(sharpness * (self.t_ear - ear) / margin)

    def mouth_yawn_score(self, mar: float, sharpness: float = 4.0) -> float:
        """Soft geometric evidence of a yawn, in [0, 1] (0.5 at T_mar)."""
        margin = max(self.t_mar - self.mar_median, 1e-6)
        return _sigmoid(sharpness * (mar - self.t_mar) / margin)

    def is_eye_closed(self, ear: float) -> bool:
        return ear < self.t_ear

    def is_yawning(self, mar: float) -> bool:
        return mar > self.t_mar

    def head_deviation(self, head_ratio: float) -> float:
        return abs(head_ratio - self.head_median)

    def is_head_off_axis(self, head_ratio: float) -> bool:
        return self.head_deviation(head_ratio) > self.t_head_dev

    # ---- persistence -------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Union[float, int, str]]) -> "DriverProfile":
        return cls(**d)  # type: ignore[arg-type]

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DriverProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        return (
            f"EAR median={self.ear_median:.3f} sigma={self.ear_sigma:.3f} -> T_ear={self.t_ear:.3f} | "
            f"MAR median={self.mar_median:.3f} sigma={self.mar_sigma:.3f} -> T_mar={self.t_mar:.3f} | "
            f"head dev limit={self.t_head_dev:.3f} | N={self.n_samples} | mode={self.mode}"
        )


# --------------------------------------------------------------------------- #
# Threshold computation (pure function -> easy to unit-test and benchmark)
# --------------------------------------------------------------------------- #
def _clean(x: np.ndarray, side: str, cfg: CalibrationConfig, floor_frac: float) -> np.ndarray:
    """Reject blink/talking outliers with a robust z-score on the relevant side."""
    med = float(np.median(x))
    sigma = max(robust_sigma(x, cfg.mad_to_sigma), floor_frac * abs(med), 1e-9)
    if side == "low":        # EAR: blinks produce strongly LOW values
        keep = x >= med - cfg.outlier_reject_k * sigma
    elif side == "high":     # MAR: talking / yawning produce strongly HIGH values
        keep = x <= med + cfg.outlier_reject_k * sigma
    else:                    # head pose: both directions
        keep = np.abs(x - med) <= cfg.outlier_reject_k * sigma
    return x[keep]


def compute_profile(
    ear: ArrayLike,
    mar: ArrayLike,
    head: ArrayLike,
    cfg: CalibrationConfig,
) -> DriverProfile:
    """Derive the personalised thresholds from raw calibration vectors.

    Raises
    ------
    CalibrationError
        if the data are too short, too noisy, or contain too many outliers (for
        example the driver kept blinking / talking / moving during the window).
    """
    ear_raw = np.asarray(ear, dtype=np.float64)
    mar_raw = np.asarray(mar, dtype=np.float64)
    head_raw = np.asarray(head, dtype=np.float64)
    n = min(ear_raw.size, mar_raw.size, head_raw.size)
    if n < cfg.min_samples_after_cleaning:
        raise CalibrationError(f"only {n} samples collected")
    if not (np.all(np.isfinite(ear_raw)) and np.all(np.isfinite(mar_raw)) and np.all(np.isfinite(head_raw))):
        raise CalibrationError("non-finite values in calibration data")

    ear_c = _clean(ear_raw, "low", cfg, cfg.sigma_floor_ear)
    mar_c = _clean(mar_raw, "high", cfg, cfg.sigma_floor_mar)
    head_c = _clean(head_raw, "both", cfg, cfg.sigma_floor_head)

    rejected_fraction = 1.0 - ear_c.size / ear_raw.size
    if rejected_fraction > cfg.max_rejected_fraction:
        raise CalibrationError(
            f"{rejected_fraction:.0%} of EAR samples were outliers (eyes closing / moving)"
        )
    # Bimodality guard: if many samples sit far ABOVE the median, the median itself lies in
    # the "eyes closed / squinting" mode, i.e. the driver did not keep their eyes open.
    ear_med0 = float(np.median(ear_raw))
    sig0 = max(robust_sigma(ear_raw, cfg.mad_to_sigma), cfg.sigma_floor_ear * abs(ear_med0), 1e-9)
    high_fraction = float(np.mean(ear_raw > ear_med0 + cfg.outlier_reject_k * sig0))
    if high_fraction > cfg.max_high_outlier_fraction:
        raise CalibrationError("eyes were not kept open during calibration")
    if min(ear_c.size, mar_c.size, head_c.size) < cfg.min_samples_after_cleaning:
        raise CalibrationError("too few samples left after outlier rejection")

    ear_med = float(np.median(ear_c))
    ear_mad = median_absolute_deviation(ear_c)
    ear_sigma = ear_mad / cfg.mad_to_sigma
    mar_med = float(np.median(mar_c))
    mar_mad = median_absolute_deviation(mar_c)
    mar_sigma = mar_mad / cfg.mad_to_sigma
    head_med = float(np.median(head_c))
    head_mad = median_absolute_deviation(head_c)
    head_sigma = head_mad / cfg.mad_to_sigma

    if ear_med <= 0.0:
        raise CalibrationError("non-positive baseline EAR")
    if ear_sigma / ear_med > cfg.max_relative_dispersion:
        raise CalibrationError(
            f"EAR too unstable (relative sigma {ear_sigma / ear_med:.2f}); keep still and look at the camera"
        )

    if cfg.mode == "ratio":
        t_ear = cfg.paper_ear_ratio * ear_med
        t_mar = cfg.paper_mar_ratio * mar_med
        t_head_dev = k_head_dev(head_med, head_sigma, cfg)
    elif cfg.mode == "mad":
        sig_e = max(ear_sigma, cfg.sigma_floor_ear * ear_med)
        t_ear = ear_med - cfg.k_ear * sig_e
        lo, hi = cfg.ear_ratio_bounds
        t_ear = float(min(max(t_ear, lo * ear_med), hi * ear_med))

        sig_m = max(mar_sigma, cfg.sigma_floor_mar * abs(mar_med))
        t_mar = mar_med + cfg.k_mar * sig_m
        t_mar = max(t_mar, cfg.paper_mar_ratio * mar_med)   # never below the paper's 140% rule
        t_head_dev = k_head_dev(head_med, head_sigma, cfg)
    else:
        raise ValueError(f"unknown calibration mode: {cfg.mode!r}")

    if not (t_mar > mar_med):
        # Degenerate (e.g. median MAR == 0): force a strictly positive margin from the data itself.
        t_mar = mar_med + max(mar_sigma, 1e-3) * cfg.k_mar

    return DriverProfile(
        ear_median=ear_med, ear_mad=ear_mad, ear_sigma=ear_sigma, t_ear=float(t_ear),
        mar_median=mar_med, mar_mad=mar_mad, mar_sigma=mar_sigma, t_mar=float(t_mar),
        head_median=head_med, head_sigma=head_sigma, t_head_dev=float(t_head_dev),
        n_samples=int(ear_c.size), mode=cfg.mode, created_at=time.time(),
    )


def k_head_dev(head_med: float, head_sigma: float, cfg: CalibrationConfig) -> float:
    """Allowed |head-ratio deviation| from the driver's own neutral posture."""
    sig = max(head_sigma, cfg.sigma_floor_head * abs(head_med), 1e-6)
    return cfg.k_head * sig


# --------------------------------------------------------------------------- #
# Online calibrator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationStatus:
    progress: float                       # 0..1
    collected: int
    target: int
    complete: bool
    message: str
    profile: Optional[DriverProfile] = None


class Calibrator:
    """Frame-by-frame collector for the start-up calibration window.

    Only frames with a detected, frontal face are counted, so the window is measured
    in *valid awake frames* (N = duration x fps), independent of wall-clock time.
    That also makes the procedure identical for live cameras and video files.
    """

    def __init__(self, cfg: CalibrationConfig) -> None:
        self.cfg = cfg
        self._ear: List[float] = []
        self._mar: List[float] = []
        self._head: List[float] = []
        self._failures = 0
        self._last_error = ""
        self._profile: Optional[DriverProfile] = None

    # ---------------------------------------------------------------- state -- #
    @property
    def is_complete(self) -> bool:
        return self._profile is not None

    @property
    def profile(self) -> Optional[DriverProfile]:
        return self._profile

    def reset(self) -> None:
        self._ear.clear()
        self._mar.clear()
        self._head.clear()
        self._failures = 0
        self._last_error = ""
        self._profile = None

    # --------------------------------------------------------------- update -- #
    def update(self, ear: float, mar: float, head_ratio: float, valid: bool) -> CalibrationStatus:
        """Feed one frame.  `valid` = face found and frontal enough for EAR/MAR."""
        target = self.cfg.target_samples
        if self._profile is not None:
            return CalibrationStatus(1.0, target, target, True, "calibrated", self._profile)

        finite = all(math.isfinite(v) for v in (ear, mar, head_ratio))
        if valid and finite:
            self._ear.append(float(ear))
            self._mar.append(float(mar))
            self._head.append(float(head_ratio))

        collected = len(self._ear)
        if collected >= target:
            try:
                self._profile = compute_profile(self._ear, self._mar, self._head, self.cfg)
                return CalibrationStatus(1.0, collected, target, True, "calibrated", self._profile)
            except CalibrationError as exc:
                self._failures += 1
                self._last_error = str(exc)
                self._ear.clear()
                self._mar.clear()
                self._head.clear()
                collected = 0

        if not valid:
            msg = "Face not found / turned away - face the camera"
        elif self._failures:
            msg = f"Restarting: {self._last_error}"
            if self._failures >= self.cfg.max_attempts_before_hint:
                msg += " | eyes open, mouth closed, head still"
        else:
            msg = "Look at the camera: eyes open, mouth closed, relaxed"
        return CalibrationStatus(collected / target, collected, target, False, msg, None)
