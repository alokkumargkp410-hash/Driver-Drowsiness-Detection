"""
config.py - Central runtime configuration for the Driver Drowsiness Detection System.

Reference
---------
Ersoy, G., Tatar, M. A., Tonbul, E., Kirbiz, S. (2026). "Improving Driver Drowsiness
Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification."
arXiv:2604.22479v1  (https://arxiv.org/html/2604.22479v1)

Design rule
-----------
This module contains NO decision thresholds on raw EAR / MAR values (for example
"EAR < 0.20").  Everything stored here is one of:

* a hardware / pipeline parameter (frame size, FPS, crop size),
* a *scaling factor* that `calibration.py` combines with per-driver statistics
  (median, MAD) to produce the driver-specific thresholds T_ear, T_mar, T_head,
* a temporal / policy parameter for the state machine (seconds, counts, PERCLOS
  fractions), which are independent of facial anatomy,
* a probability-domain parameter for CNN fusion.

Values are grouped in frozen dataclasses so they cannot be mutated accidentally at
runtime.  To experiment, create a modified copy with `dataclasses.replace(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
PROFILES_DIR: Path = PROJECT_ROOT / "profiles"


# --------------------------------------------------------------------------- #
# Video / capture
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VideoConfig:
    """Capture and processing resolution."""

    frame_width: int = 640
    frame_height: int = 480
    target_fps: float = 30.0
    camera_index: int = 0
    capture_buffer_size: int = 1          # minimise latency on webcams
    mirror_webcam: bool = True            # selfie-view for live cameras only
    max_consecutive_read_failures: int = 30
    fps_ema_alpha: float = 0.1            # smoothing for the FPS read-out


# --------------------------------------------------------------------------- #
# Landmark extraction (MediaPipe Face Mesh, 468/478-point topology)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LandmarkConfig:
    """MediaPipe Face Mesh settings and landmark index maps.

    The paper (Sec. III-A) extracts 468 landmarks with MediaPipe Face Mesh and
    evaluates EAR (Eq. 1) on six points per eye and MAR (Eq. 3) on the inner-lip
    points.  The index tuples below select the Face Mesh vertices that correspond
    to the classic 68-point p1..p6 eye layout and the inner-lip layout.
    """

    # Tasks-API model (used when `mediapipe.solutions` is unavailable).
    model_path: Path = MODELS_DIR / "face_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    auto_download_model: bool = True
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Temporal landmark stabiliser: weight of the NEW observation (1.0 = no smoothing).
    landmark_ema_alpha: float = 0.65

    # Face Mesh indices, ordered p1..p6 (Fig. 1 of the paper):
    # p1 = outer corner, p2/p3 = upper lid, p4 = inner corner, p5/p6 = lower lid.
    right_eye: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    left_eye: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

    # Inner-lip vertices for MAR: mouth corners and three upper/lower pairs.
    mouth_corners: Tuple[int, int] = (78, 308)
    mouth_vertical_pairs: Tuple[Tuple[int, int], ...] = ((82, 87), (13, 14), (312, 317))

    # Outer-lip vertices, used only to size the mouth crop for the CNN.
    mouth_outer: Tuple[int, ...] = (61, 291, 0, 17, 37, 267, 84, 314, 39, 269, 91, 321)

    # Head-pose proxy landmarks.
    nose_tip: int = 1
    face_left: int = 234
    face_right: int = 454
    outer_eye_corners: Tuple[int, int] = (33, 263)

    # Quality gate (NOT a drowsiness threshold): frames whose horizontal nose offset
    # from the face midline exceeds this fraction are considered "turned away" and
    # their EAR/MAR are not trusted; the CNN is used alone for those frames.
    yaw_valid_tolerance: float = 0.22


# --------------------------------------------------------------------------- #
# CNN crops
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CropConfig:
    """Eye / mouth patch extraction (paper: 64x64 crops + CLAHE)."""

    eye_crop_size: Tuple[int, int] = (64, 64)       # (width, height)
    mouth_crop_size: Tuple[int, int] = (64, 64)
    eye_box_scale: float = 1.9      # square side = scale * eye width
    mouth_box_scale: float = 1.6    # square side = scale * max(lip width, lip height)
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (4, 4)


# --------------------------------------------------------------------------- #
# Calibration (core contribution)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationConfig:
    """Person-specific baseline calibration.

    mode = "mad"   : T_ear = median(EAR) - k_ear * sigma_ear,
                     T_mar = median(MAR) + k_mar * sigma_mar,
                     sigma = MAD / 0.6745  (robust std estimate).
    mode = "ratio" : paper-faithful variant (Sec. IV-B): T_ear = 0.75 * baseline EAR,
                     T_mar = 1.40 * baseline MAR (baseline = median here).
    """

    mode: str = "mad"                    # "mad" | "ratio"
    duration_seconds: float = 5.0        # paper: 5 s neutral-posture calibration
    assumed_fps: float = 30.0            # converts seconds -> required valid frames
    mad_to_sigma: float = 0.6745         # MAD / 0.6745 ~= sigma for Gaussian data

    # Robust-scale multipliers k.
    k_ear: float = 5.0
    k_mar: float = 6.0
    k_head: float = 5.0

    # Lower bound for sigma, expressed as a FRACTION of the driver's own median.
    # Prevents a degenerate threshold sitting on the median when the driver is very
    # still (MAD -> 0).  k_ear * floor = 0.25 reproduces the paper's 75% rule as the
    # minimum margin.
    sigma_floor_ear: float = 0.05
    sigma_floor_mar: float = 0.15
    sigma_floor_head: float = 0.03

    # Paper ratio rule (also used as safety clamps in "mad" mode).
    paper_ear_ratio: float = 0.75
    paper_mar_ratio: float = 1.40
    ear_ratio_bounds: Tuple[float, float] = (0.55, 0.90)   # T_ear / median(EAR)

    # Blink / talking rejection inside the calibration window (in sigma units).
    outlier_reject_k: float = 3.5
    max_rejected_fraction: float = 0.45
    max_high_outlier_fraction: float = 0.20   # median stuck in the low mode (eyes mostly closed)
    max_relative_dispersion: float = 0.25   # sigma_ear / median_ear on cleaned data
    min_samples_after_cleaning: int = 40

    # Fixed number of consecutive failed attempts before the message changes.
    max_attempts_before_hint: int = 2

    @property
    def target_samples(self) -> int:
        """Number of valid (face found, frontal) frames required: N = T * fps."""
        return max(1, int(round(self.duration_seconds * self.assumed_fps)))


# --------------------------------------------------------------------------- #
# CNN classifier & fusion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CNNConfig:
    """Two lightweight CNNs (eye open/closed, mouth neutral/yawn), 64x64 input."""

    eye_weights: Path = MODELS_DIR / "eye_cnn.pt"
    mouth_weights: Path = MODELS_DIR / "mouth_cnn.pt"
    enabled: bool = True
    device: str = "cpu"
    num_threads: int = 2
    inference_stride: int = 2            # run CNN every N frames, reuse in between
    width: int = 16                      # base channel width of the network
    dropout: float = 0.2

    # Probability-domain fusion (class "closed" / "yawn" = positive class).
    geometry_weight_eye: float = 0.45    # weight of calibrated-EAR score in fusion
    geometry_weight_mouth: float = 0.45
    geometry_sigmoid_sharpness: float = 4.0   # score = 0.5 at the threshold; 0.018 at baseline
    decision_enter_prob: float = 0.60    # hysteresis: switch ON above this
    decision_exit_prob: float = 0.40     # hysteresis: switch OFF below this


# --------------------------------------------------------------------------- #
# Temporal state machine (60 s sliding window)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateConfig:
    """Sliding-window fatigue evaluation."""

    window_seconds: float = 60.0
    eye_smoothing_frames: int = 3        # median filter: removes landmark jitter, keeps blinks
    mar_smoothing_frames: int = 15       # paper Sec. III-C: 15-frame buffer
    max_frame_gap_s: float = 0.25        # clip dt after stalls

    # Eye-closure event classification (seconds).
    blink_min_duration_s: float = 0.06
    micro_sleep_min_duration_s: float = 0.50

    # Yawn event.
    yawn_min_duration_s: float = 1.50
    yawn_refractory_s: float = 3.00

    # Head off-axis (nodding) sustained duration.
    head_off_axis_min_duration_s: float = 1.50

    # Level-1 (WARNING) triggers.
    perclos_min_span_s: float = 10.0     # do not trust PERCLOS before this much data
    warn_perclos: float = 0.15
    warn_microsleeps: int = 1
    warn_yawns: int = 2
    warn_blinks_per_window: int = 40
    face_lost_alert_s: float = 3.0

    # Level-2 (CRITICAL) triggers.
    critical_perclos: float = 0.30
    critical_microsleeps: int = 2
    critical_yawns: int = 4
    critical_closure_s: float = 1.50     # a single continuous closure this long
    critical_breach_count: int = 2       # simultaneous Level-1 breaches

    alert_clear_s: float = 3.0           # hold time before the level may drop
    face_loss_grace_s: float = 0.50      # keep episodes alive through short dropouts


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HUDConfig:
    graph_history_frames: int = 150
    graph_height: int = 70
    graph_width: int = 210
    font_scale: float = 0.5
    panel_alpha: float = 0.55
    window_name: str = "Driver Drowsiness Monitor (Ersoy et al., 2026)"


# --------------------------------------------------------------------------- #
# Aggregated defaults
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    crops: CropConfig = field(default_factory=CropConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    state: StateConfig = field(default_factory=StateConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)


DEFAULT_CONFIG: AppConfig = AppConfig()
