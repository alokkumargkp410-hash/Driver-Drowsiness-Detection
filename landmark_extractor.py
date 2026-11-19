"""
landmark_extractor.py - Real-time facial landmarks, EAR / MAR and CNN patches.

Backend
-------
MediaPipe Face Mesh (468/478 landmarks, as in Sec. III-A of Ersoy et al., 2026).
Two MediaPipe APIs are supported and chosen automatically:

* legacy ``mediapipe.solutions.face_mesh``   (older mediapipe releases), or
* the Tasks API ``FaceLandmarker``           (current releases; needs the
  ``face_landmarker.task`` model file, downloaded on first use if missing).

The six-point eye layout (p1..p6) and the inner-lip layout of the classic 68-point
scheme are mapped to the corresponding Face Mesh vertices (see config.LandmarkConfig).

Per frame this module returns
* EAR (Eq. 1) per eye and averaged,
* MAR (Eq. 3 variant, three vertical pairs for lower jitter),
* a head-pose proxy (nose-to-eye-line distance / inter-ocular distance) and a yaw
  offset used only as a quality gate,
* CLAHE-enhanced, fixed-size BGR crops of both eyes and the mouth for the CNN.

Landmark loss is handled gracefully: `face_found=False` is reported, the temporal
stabiliser is reset, and the first frame after re-acquisition is flagged.
"""
from __future__ import annotations

import logging
import math
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

from config import CropConfig, LandmarkConfig

LOG = logging.getLogger(__name__)

Point = np.ndarray


# --------------------------------------------------------------------------- #
# Pure geometry (unit-testable without any camera / model)
# --------------------------------------------------------------------------- #
def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(p: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)   (Eq. 1).  `p` has shape (6, 2)."""
    horiz = _dist(p[0], p[3])
    if horiz < 1e-6:
        return float("nan")
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horiz)


def mouth_aspect_ratio(lm: np.ndarray, corners: Tuple[int, int],
                       pairs: Tuple[Tuple[int, int], ...]) -> float:
    """MAR = mean(vertical inner-lip openings) / mouth width.

    Generalises Eq. 3 (two vertical pairs) to the three upper/lower inner-lip pairs
    available in the Face Mesh topology.
    """
    width = _dist(lm[corners[0]], lm[corners[1]])
    if width < 1e-6:
        return float("nan")
    vert = sum(_dist(lm[u], lm[l]) for u, l in pairs) / len(pairs)
    return vert / width


def head_pose_proxy(lm: np.ndarray, cfg: LandmarkConfig) -> Tuple[float, float]:
    """Return (head_ratio, yaw_offset).

    head_ratio : vertical distance nose-tip <-> eye line, divided by the inter-ocular
                 distance (scale invariant).  The paper monitors this distance to detect
                 downward head tilt; the calibrated baseline makes it person-specific.
    yaw_offset : |nose_x position within the face width - 0.5|; 0 = frontal.
    """
    a, b = cfg.outer_eye_corners
    iod = _dist(lm[a], lm[b])
    if iod < 1e-6:
        return float("nan"), float("nan")
    eye_line_y = 0.5 * (lm[a][1] + lm[b][1])
    head_ratio = (lm[cfg.nose_tip][1] - eye_line_y) / iod
    left_x, right_x = lm[cfg.face_left][0], lm[cfg.face_right][0]
    width = abs(right_x - left_x)
    if width < 1e-6:
        return head_ratio, float("nan")
    yaw = abs((lm[cfg.nose_tip][0] - min(left_x, right_x)) / width - 0.5)
    return float(head_ratio), float(yaw)


def crop_square(frame: np.ndarray, center: Tuple[float, float], side: float,
                out_size: Tuple[int, int]) -> np.ndarray:
    """Crop a square region (border-replicated) and resize it in a single warpAffine."""
    side = max(float(side), 4.0)
    sx = out_size[0] / side
    sy = out_size[1] / side
    tx = -(center[0] - side / 2.0) * sx
    ty = -(center[1] - side / 2.0) * sy
    m = np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float32)
    return cv2.warpAffine(frame, m, out_size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class FrameFeatures:
    """Everything the downstream pipeline needs from one frame."""

    face_found: bool = False
    geometry_valid: bool = False          # frontal enough for trustworthy EAR / MAR
    reacquired: bool = False              # first detected frame after a loss
    lost_frames: int = 0                  # consecutive frames without a face
    ear_left: float = float("nan")
    ear_right: float = float("nan")
    ear: float = float("nan")
    mar: float = float("nan")
    head_ratio: float = float("nan")
    yaw_offset: float = float("nan")
    landmarks_px: Optional[np.ndarray] = None
    eye_patches: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (right, left) uint8 BGR
    mouth_patch: Optional[np.ndarray] = None                      # uint8 BGR
    face_bbox: Optional[Tuple[int, int, int, int]] = None


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _Backend(Protocol):
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        """Return normalised landmarks (N, 3) or None."""

    def close(self) -> None: ...


class _LegacyBackend:
    """mediapipe.solutions.face_mesh (available in older mediapipe releases)."""

    def __init__(self, cfg: LandmarkConfig) -> None:
        import mediapipe as mp  # local import keeps the module importable without mediapipe

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.num_faces,
            refine_landmarks=True,
            min_detection_confidence=cfg.min_face_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        rgb.flags.writeable = False
        res = self._mesh.process(rgb)
        rgb.flags.writeable = True
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0].landmark
        return np.array([(l.x, l.y, l.z) for l in lms], dtype=np.float32)

    def close(self) -> None:
        self._mesh.close()


class _TasksBackend:
    """mediapipe.tasks FaceLandmarker in VIDEO mode (temporal tracking)."""

    def __init__(self, cfg: LandmarkConfig) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        path = cfg.model_path
        if not path.exists():
            if not cfg.auto_download_model:
                raise FileNotFoundError(f"FaceLandmarker model not found: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            LOG.info("Downloading FaceLandmarker model to %s", path)
            try:
                urllib.request.urlretrieve(cfg.model_url, str(path))
            except Exception as exc:  # network errors, proxies, ...
                if path.exists():
                    path.unlink()
                raise RuntimeError(
                    f"Could not download the MediaPipe face model ({exc}).\n"
                    f"Download it manually from:\n  {cfg.model_url}\nand save it as:\n  {path}"
                ) from exc
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=cfg.num_faces,
            min_face_detection_confidence=cfg.min_face_detection_confidence,
            min_face_presence_confidence=cfg.min_face_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        ts = max(int(timestamp_ms), self._last_ts + 1)   # must be strictly increasing
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect_for_video(image, ts)
        if not res.face_landmarks:
            return None
        return np.array([(l.x, l.y, l.z) for l in res.face_landmarks[0]], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()


def create_backend(cfg: LandmarkConfig) -> _Backend:
    """Pick the legacy solution if present, else the Tasks API."""
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions"), "face_mesh"):
        return _LegacyBackend(cfg)
    return _TasksBackend(cfg)


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class LandmarkExtractor:
    """Frame -> FrameFeatures.  Stateful (temporal landmark stabiliser, loss tracking)."""

    def __init__(self, lm_cfg: LandmarkConfig, crop_cfg: CropConfig,
                 backend: Optional[_Backend] = None) -> None:
        self.lm_cfg = lm_cfg
        self.crop_cfg = crop_cfg
        self._backend: _Backend = backend if backend is not None else create_backend(lm_cfg)
        self._clahe = cv2.createCLAHE(clipLimit=crop_cfg.clahe_clip_limit,
                                      tileGridSize=crop_cfg.clahe_tile_grid)
        self._prev: Optional[np.ndarray] = None
        self._lost = 0
        self._ever_found = False
        self._errors = 0

    # ------------------------------------------------------------------ API -- #
    def process(self, frame_bgr: np.ndarray, timestamp_s: float,
                want_patches: bool = True) -> FrameFeatures:
        h, w = frame_bgr.shape[:2]
        norm: Optional[np.ndarray] = None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            norm = self._backend.detect(rgb, int(timestamp_s * 1000.0))
        except Exception as exc:     # never let a tracker hiccup kill the pipeline
            self._errors += 1
            if self._errors <= 3 or self._errors % 100 == 0:
                LOG.warning("Landmark backend error (%d): %s", self._errors, exc)
            norm = None

        if norm is None:
            self._lost += 1
            self._prev = None            # drop stale history so re-acquisition starts clean
            return FrameFeatures(face_found=False, lost_frames=self._lost)

        reacquired = self._ever_found and self._lost > 0
        self._lost = 0
        self._ever_found = True
        return self._features_from_landmarks(frame_bgr, norm, w, h, reacquired, want_patches)

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception:    # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------- internals -- #
    def _features_from_landmarks(self, frame: np.ndarray, norm: np.ndarray, w: int, h: int,
                                 reacquired: bool, want_patches: bool) -> FrameFeatures:
        cfg = self.lm_cfg
        px = norm[:, :2].astype(np.float32) * np.array([w, h], dtype=np.float32)
        if self._prev is not None and self._prev.shape == px.shape:
            a = cfg.landmark_ema_alpha
            px = a * px + (1.0 - a) * self._prev       # jitter suppression
        self._prev = px

        right = px[list(cfg.right_eye)]
        left = px[list(cfg.left_eye)]
        ear_r = eye_aspect_ratio(right)
        ear_l = eye_aspect_ratio(left)
        ear = float(np.nanmean([ear_r, ear_l])) if not (math.isnan(ear_r) and math.isnan(ear_l)) else float("nan")
        mar = mouth_aspect_ratio(px, cfg.mouth_corners, cfg.mouth_vertical_pairs)
        head_ratio, yaw = head_pose_proxy(px, cfg)
        valid = (not math.isnan(ear) and not math.isnan(mar) and not math.isnan(yaw)
                 and yaw <= cfg.yaw_valid_tolerance)

        x0, y0 = np.floor(px.min(axis=0)).astype(int)
        x1, y1 = np.ceil(px.max(axis=0)).astype(int)
        bbox = (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w - 1)), int(min(y1, h - 1)))

        eye_patches = None
        mouth_patch = None
        if want_patches:
            eye_patches = (self._eye_patch(frame, right), self._eye_patch(frame, left))
            mouth_patch = self._mouth_patch(frame, px)

        return FrameFeatures(
            face_found=True, geometry_valid=bool(valid), reacquired=reacquired, lost_frames=0,
            ear_left=ear_l, ear_right=ear_r, ear=ear, mar=mar,
            head_ratio=head_ratio, yaw_offset=yaw, landmarks_px=px,
            eye_patches=eye_patches, mouth_patch=mouth_patch, face_bbox=bbox,
        )

    def _enhance(self, patch: np.ndarray) -> np.ndarray:
        if not self.crop_cfg.use_clahe:
            return patch
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _eye_patch(self, frame: np.ndarray, eye_pts: np.ndarray) -> np.ndarray:
        center = eye_pts.mean(axis=0)
        width = _dist(eye_pts[0], eye_pts[3])
        patch = crop_square(frame, (float(center[0]), float(center[1])),
                            width * self.crop_cfg.eye_box_scale, self.crop_cfg.eye_crop_size)
        return self._enhance(patch)

    def _mouth_patch(self, frame: np.ndarray, px: np.ndarray) -> np.ndarray:
        pts = px[list(self.lm_cfg.mouth_outer)]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = (lo + hi) / 2.0
        side = max(hi[0] - lo[0], hi[1] - lo[1]) * self.crop_cfg.mouth_box_scale
        patch = crop_square(frame, (float(center[0]), float(center[1])), side,
                            self.crop_cfg.mouth_crop_size)
        return self._enhance(patch)
