"""
Unit tests (no camera and no MediaPipe model needed).

    python -m unittest discover -s tests -v          (run from the project root)
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration import (CalibrationError, Calibrator, DriverProfile, compute_profile,  # noqa: E402
                         median_absolute_deviation, robust_sigma)
from cnn_classifier import (CNNStateClassifier, HysteresisDecision, StateCNN,  # noqa: E402
                            fuse_probabilities, patches_to_tensor, save_checkpoint)
from config import DEFAULT_CONFIG, CNNConfig, LandmarkConfig  # noqa: E402
from landmark_extractor import (FrameFeatures, LandmarkExtractor, crop_square,  # noqa: E402
                                eye_aspect_ratio, head_pose_proxy, mouth_aspect_ratio)
from main import DrowsinessPipeline  # noqa: E402
from state_machine import DrowsinessStateMachine, FrameObservation  # noqa: E402

CFG = DEFAULT_CONFIG
RNG = np.random.default_rng(0)


def calib_vectors(ear_mu=0.30, n=150, blink_frac=0.08, mar_mu=0.03):
    ear = RNG.normal(ear_mu, 0.012, n)
    blink = RNG.random(n) < blink_frac
    ear[blink] = ear_mu * RNG.uniform(0.3, 0.6, blink.sum())
    mar = np.abs(RNG.normal(mar_mu, 0.004, n))
    head = RNG.normal(0.8, 0.01, n)
    return ear, mar, head


class StatsTests(unittest.TestCase):
    def test_mad_and_sigma(self):
        x = np.array([1, 2, 3, 4, 100.0])
        self.assertAlmostEqual(median_absolute_deviation(x), 1.0)
        self.assertAlmostEqual(robust_sigma(x), 1.0 / 0.6745)

    def test_mad_matches_gaussian_sigma(self):
        x = RNG.normal(0, 2.0, 200000)
        self.assertAlmostEqual(robust_sigma(x), 2.0, delta=0.05)


class GeometryTests(unittest.TestCase):
    def test_ear_open_vs_closed(self):
        open_eye = np.array([[0, 0], [3, -2], [7, -2], [10, 0], [7, 2], [3, 2]], float)
        closed = open_eye.copy()
        closed[[1, 2], 1] = -0.2
        closed[[4, 5], 1] = 0.2
        self.assertAlmostEqual(eye_aspect_ratio(open_eye), 0.4, places=6)
        self.assertLess(eye_aspect_ratio(closed), 0.1)

    def test_ear_scale_invariant(self):
        p = np.array([[0, 0], [3, -2], [7, -2], [10, 0], [7, 2], [3, 2]], float)
        self.assertAlmostEqual(eye_aspect_ratio(p), eye_aspect_ratio(p * 7.3))

    def test_mar(self):
        lm = np.zeros((478, 2))
        cfg = LandmarkConfig()
        lm[78], lm[308] = (0, 0), (10, 0)
        for (u, l), x in zip(cfg.mouth_vertical_pairs, (3, 5, 7)):
            lm[u], lm[l] = (x, -2), (x, 2)
        self.assertAlmostEqual(mouth_aspect_ratio(lm, cfg.mouth_corners, cfg.mouth_vertical_pairs), 0.4)

    def test_head_proxy(self):
        cfg = LandmarkConfig()
        lm = np.zeros((478, 2))
        lm[33], lm[263] = (0, 0), (10, 0)
        lm[cfg.nose_tip] = (5, 6)
        lm[cfg.face_left], lm[cfg.face_right] = (-2, 3), (12, 3)
        ratio, yaw = head_pose_proxy(lm, cfg)
        self.assertAlmostEqual(ratio, 0.6)
        self.assertLess(yaw, 0.01)

    def test_crop_shape(self):
        img = RNG.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        out = crop_square(img, (5, 5), 40, (64, 64))     # partially outside the frame
        self.assertEqual(out.shape, (64, 64, 3))


class CalibrationTests(unittest.TestCase):
    def test_threshold_follows_anatomy(self):
        p_wide = compute_profile(*calib_vectors(0.34), CFG.calibration)
        p_narrow = compute_profile(*calib_vectors(0.20), CFG.calibration)
        self.assertGreater(p_wide.t_ear, p_narrow.t_ear)
        for p in (p_wide, p_narrow):
            self.assertLess(p.t_ear, p.ear_median)
            self.assertGreater(p.t_mar, p.mar_median)
            lo, hi = CFG.calibration.ear_ratio_bounds
            self.assertGreaterEqual(p.t_ear / p.ear_median, lo - 1e-9)
            self.assertLessEqual(p.t_ear / p.ear_median, hi + 1e-9)

    def test_blinks_do_not_bias_baseline(self):
        ear, mar, head = calib_vectors(0.30, blink_frac=0.10)
        p = compute_profile(ear, mar, head, CFG.calibration)
        self.assertAlmostEqual(p.ear_median, 0.30, delta=0.01)

    def test_formula_when_unclamped(self):
        # sigma large enough to exceed the floor so the pure formula applies
        ear = RNG.normal(0.30, 0.02, 400)
        p = compute_profile(ear, np.full(400, 0.03) + RNG.normal(0, 0.002, 400), RNG.normal(0.8, 0.01, 400),
                            replace(CFG.calibration, outlier_reject_k=10.0))
        expected = p.ear_median - CFG.calibration.k_ear * max(p.ear_sigma, 0.05 * p.ear_median)
        lo, hi = CFG.calibration.ear_ratio_bounds
        expected = min(max(expected, lo * p.ear_median), hi * p.ear_median)
        self.assertAlmostEqual(p.t_ear, expected, places=9)

    def test_ratio_mode_is_paper_rule(self):
        cfg = replace(CFG.calibration, mode="ratio")
        p = compute_profile(*calib_vectors(0.30), cfg)
        self.assertAlmostEqual(p.t_ear, 0.75 * p.ear_median)
        self.assertAlmostEqual(p.t_mar, 1.40 * p.mar_median)

    def test_rejects_bad_windows(self):
        with self.assertRaises(CalibrationError):
            compute_profile(*[np.ones(5)] * 3, CFG.calibration)
        ear = np.where(RNG.random(150) < 0.7, 0.08, 0.30) + RNG.normal(0, 0.008, 150)   # eyes mostly closed
        with self.assertRaises(CalibrationError):
            compute_profile(ear, np.full(150, 0.03), np.full(150, 0.8), CFG.calibration)

    def test_online_calibrator_and_persistence(self):
        cal = Calibrator(CFG.calibration)
        ear, mar, head = calib_vectors(0.28)
        status = None
        for i in range(len(ear)):
            status = cal.update(ear[i], mar[i], head[i], True)
        self.assertTrue(status.complete)
        self.assertIsNotNone(status.profile)
        with tempfile.TemporaryDirectory() as d:
            path = status.profile.save(Path(d) / "p.json")
            loaded = DriverProfile.load(path)
        self.assertEqual(loaded, status.profile)

    def test_calibrator_ignores_invalid_frames(self):
        cal = Calibrator(CFG.calibration)
        for _ in range(50):
            st = cal.update(float("nan"), float("nan"), float("nan"), False)
        self.assertEqual(st.collected, 0)
        self.assertFalse(st.complete)

    def test_scores(self):
        p = compute_profile(*calib_vectors(0.30), CFG.calibration)
        self.assertAlmostEqual(p.eye_closed_score(p.t_ear), 0.5, places=6)
        self.assertLess(p.eye_closed_score(p.ear_median), 0.05)
        self.assertGreater(p.eye_closed_score(0.05), 0.95)
        self.assertAlmostEqual(p.mouth_yawn_score(p.t_mar), 0.5, places=6)
        self.assertGreater(p.mouth_yawn_score(p.t_mar * 5), 0.95)


class StateMachineTests(unittest.TestCase):
    def run_seq(self, sm, segments, fps=30.0, t0=0.0):
        """segments: list of (duration_s, closed, yawning, head_off)."""
        t, status = t0, None
        for dur, closed, yawn, head in segments:
            for _ in range(int(dur * fps)):
                t += 1.0 / fps
                status = sm.update(FrameObservation(t, True, closed, yawn, head))
        return status, t

    def test_blinks_vs_microsleep(self):
        sm = DrowsinessStateMachine(CFG.state)
        segs = []
        for _ in range(5):
            segs += [(0.8, False, False, False), (0.15, True, False, False)]
        segs += [(0.5, False, False, False)]
        st, t = self.run_seq(sm, segs)
        self.assertEqual(st.metrics.blink_count, 5)
        self.assertEqual(st.metrics.microsleep_count, 0)
        self.assertEqual(st.level, 0)
        st, t = self.run_seq(sm, [(0.7, True, False, False)], t0=t)
        self.assertEqual(st.metrics.microsleep_count, 1)
        self.assertEqual(st.level, 1)

    def test_critical_continuous_closure(self):
        sm = DrowsinessStateMachine(CFG.state)
        st, _ = self.run_seq(sm, [(2.0, False, False, False), (1.8, True, False, False)])
        self.assertEqual(st.level, 2)

    def test_yawn_count_and_levels(self):
        sm = DrowsinessStateMachine(CFG.state)
        segs = []
        for _ in range(2):
            segs += [(3.0, False, False, False), (2.0, False, True, False)]
        st, _ = self.run_seq(sm, segs)
        self.assertEqual(st.metrics.yawn_count, 2)
        self.assertEqual(st.level, 1)

    def test_fused_breaches_go_critical(self):
        sm = DrowsinessStateMachine(CFG.state)
        segs = [(1.0, False, False, False)]
        for _ in range(2):
            segs += [(1.0, False, False, False), (2.0, False, True, False)]
        segs += [(2.0, False, False, True)]                # head nod while 2 yawns are in window
        st, _ = self.run_seq(sm, segs)
        self.assertEqual(st.level, 2)

    def test_window_expiry_and_hysteresis(self):
        sm = DrowsinessStateMachine(CFG.state)
        st, t = self.run_seq(sm, [(1.0, False, False, False), (0.8, True, False, False)])
        self.assertEqual(st.level, 1)
        st, t = self.run_seq(sm, [(1.0, False, False, False)], t0=t)
        self.assertEqual(st.level, 1)                      # still held (alert_clear_s = 3 s)
        st, t = self.run_seq(sm, [(70.0, False, False, False)], t0=t)
        self.assertEqual(st.level, 0)
        self.assertEqual(st.metrics.microsleep_count, 0)

    def test_perclos(self):
        sm = DrowsinessStateMachine(CFG.state)
        segs = []
        for _ in range(13):
            segs += [(0.8, False, False, False), (0.2, True, False, False)]   # 20% closed
        st, _ = self.run_seq(sm, segs)
        self.assertAlmostEqual(st.metrics.perclos, 0.2, delta=0.03)
        self.assertGreaterEqual(st.level, 1)

    def test_face_loss(self):
        sm = DrowsinessStateMachine(CFG.state)
        st, t = self.run_seq(sm, [(1.0, False, False, False)])
        for _ in range(int(4 * 30)):
            t += 1 / 30
            st = sm.update(FrameObservation(t, False, False, False))
        self.assertEqual(st.level, 1)
        self.assertIn("DRIVER NOT VISIBLE", st.reasons)


class CNNTests(unittest.TestCase):
    def test_forward_and_size(self):
        m = StateCNN().eval()
        out = m(patches_to_tensor([RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)] * 2))
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertLess(sum(p.numel() for p in m.parameters()), 100_000)

    def test_classifier_with_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_checkpoint(StateCNN(), d / "e.pt", "eye", ("open", "closed"), 16)
            save_checkpoint(StateCNN(), d / "m.pt", "mouth", ("neutral", "yawning"), 16)
            cfg = replace(CNNConfig(), eye_weights=d / "e.pt", mouth_weights=d / "m.pt", inference_stride=2)
            clf = CNNStateClassifier(cfg)
            self.assertTrue(clf.eye_available and clf.mouth_available)
            patch = RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            o1 = clf.predict([patch, patch], patch)
            o2 = clf.predict([patch, patch], patch)
            self.assertTrue(o1.fresh)
            self.assertFalse(o2.fresh)                     # stride re-uses the cached result
            self.assertTrue(0.0 <= o1.eye_closed_prob <= 1.0 and 0.0 <= o1.yawn_prob <= 1.0)

    def test_missing_weights_is_graceful(self):
        cfg = replace(CNNConfig(), eye_weights=Path("/nonexistent/a.pt"), mouth_weights=Path("/nonexistent/b.pt"))
        self.assertFalse(CNNStateClassifier(cfg).available)

    def test_fusion_and_hysteresis(self):
        self.assertEqual(fuse_probabilities(0.9, 0.1, 0.5).source, "fused")
        self.assertAlmostEqual(fuse_probabilities(0.9, 0.1, 0.5).probability, 0.5)
        self.assertEqual(fuse_probabilities(None, 0.8, 0.5).source, "cnn")
        self.assertEqual(fuse_probabilities(0.8, None, 0.5).source, "geometry")
        self.assertEqual(fuse_probabilities(None, None, 0.5).source, "none")
        h = HysteresisDecision(0.6, 0.4)
        self.assertEqual([h.update(p) for p in (0.5, 0.7, 0.5, 0.45, 0.3)], [False, True, True, True, False])


class _NoFaceBackend:
    def detect(self, rgb, ts):
        return None

    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def feats(self, ear, mar=0.03, head=0.8, valid=True):
        return FrameFeatures(face_found=True, geometry_valid=valid, ear=ear, mar=mar, head_ratio=head,
                             yaw_offset=0.0)

    def test_end_to_end_without_cnn(self):
        cfg = replace(CFG, cnn=replace(CFG.cnn, enabled=False))
        pipe = DrowsinessPipeline(cfg, use_cnn=False, extractor=LandmarkExtractor(
            cfg.landmarks, cfg.crops, backend=_NoFaceBackend()))
        ear, mar, head = calib_vectors(0.27)
        t = 0.0
        for i in range(len(ear)):
            t += 1 / 30
            res = pipe.process_features(self.feats(ear[i], mar[i], head[i]), t)
        self.assertEqual(pipe.phase, "TRACKING")
        prof = pipe.profile
        self.assertAlmostEqual(prof.ear_median, 0.27, delta=0.01)

        for _ in range(60):                                 # relaxed open eyes
            t += 1 / 30
            res = pipe.process_features(self.feats(prof.ear_median), t)
        self.assertFalse(res.eye_closed)
        self.assertEqual(res.alert.level, 0)

        for _ in range(60):                                 # 2 s eye closure -> critical
            t += 1 / 30
            res = pipe.process_features(self.feats(prof.t_ear * 0.5), t)
        self.assertTrue(res.eye_closed)
        self.assertEqual(res.alert.level, 2)

    def test_untrusted_geometry_without_cnn_keeps_state(self):
        cfg = replace(CFG, cnn=replace(CFG.cnn, enabled=False))
        pipe = DrowsinessPipeline(cfg, use_cnn=False, extractor=LandmarkExtractor(
            cfg.landmarks, cfg.crops, backend=_NoFaceBackend()))
        ear, mar, head = calib_vectors(0.27)
        for i in range(len(ear)):
            pipe.process_features(self.feats(ear[i], mar[i], head[i]), i / 30)
        res = pipe.process_features(self.feats(0.01, valid=False), 6.0)   # garbage EAR from head turn
        self.assertFalse(res.eye_closed)
        self.assertEqual(res.eye_source, "none")

    def test_extractor_handles_no_face(self):
        ext = LandmarkExtractor(CFG.landmarks, CFG.crops, backend=_NoFaceBackend())
        f = ext.process(np.zeros((480, 640, 3), np.uint8), 0.0)
        self.assertFalse(f.face_found)
        self.assertEqual(ext.process(np.zeros((480, 640, 3), np.uint8), 0.03).lost_frames, 2)


if __name__ == "__main__":
    unittest.main()
