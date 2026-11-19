# Personalised Driver Drowsiness Detection

Real-time driver drowsiness detection with **driver-specific EAR / MAR thresholds** computed from a short
start-up calibration (median + MAD statistics), optional **CNN validation** of eye and mouth state, and a
**60-second sliding-window state machine** that raises Level 1 (Warning) and Level 2 (Critical) alerts.

> Implementation of the ideas in Ersoy, Tatar, Tonbul, Kırbız (2026), *"Improving Driver Drowsiness
> Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification"*,
> arXiv:2604.22479v1 — <https://arxiv.org/html/2604.22479v1?hl=en-GB>.

---

## 1. Abstract

Fixed Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) cut-offs such as `EAR < 0.20` do not generalise across
people: eye opening, lip thickness and mouth width vary with anatomy, camera distance and lighting. This project
removes every hard-coded decision value on EAR/MAR. During the first ~5 s of operation (150 valid frames) the
system measures the driver's own neutral baseline and derives `T_ear`, `T_mar` and a head-pose limit from robust
statistics (median, MAD). Per-frame decisions blend this calibrated geometric evidence with the output of two
small CNNs (eye open/closed, mouth neutral/yawning) and feed a temporal engine that counts blinks `N_b(t)`,
micro-sleeps `N_s(t)`, yawns `N_y(t)` and PERCLOS over a 60-second window.

## 2. Academic rationale

Ersoy et al. identify three problems with classical landmark-based monitoring and with the usual
"EAR < constant" implementation:

1. **Fixed thresholds do not transfer between people** (facial structure, eye shape, mouth geometry, illumination).
2. **Landmark geometry alone is fragile** under head rotation, shadows and jitter, whereas CNNs on eye / mouth
   crops are more robust but costlier.
3. **Single-frame evidence is noisy**, so blinks, talking and brief mouth openings must be filtered temporally.

How each is addressed here:

| Paper element (section) | Where in this code |
|---|---|
| MediaPipe Face Mesh landmarks, EAR Eq. 1, MAR Eq. 3 (III-A) | `landmark_extractor.py` |
| Short personalised calibration before driving (III-A, IV-B: 5 s, neutral posture) | `calibration.py`, `CalibrationConfig.duration_seconds = 5.0` |
| Personalised EAR/MAR thresholds (IV-B: 75 % of baseline EAR, 140 % of baseline MAR) | `calibration.py`, `mode="ratio"` |
| Head-down detection from nose-tip / eye-level distance (III-A) | `landmark_extractor.head_pose_proxy` + calibrated `t_head_dev` |
| Two CNNs on 64×64 crops, CLAHE, flip / brightness / rotation augmentation (III-B, IV) | `cnn_classifier.py`, `train_cnn.py`, `landmark_extractor.py` |
| 15-frame smoothing buffer, identical warning logic for all methods (III-C) | `state_machine.RollingSmoother`, `config.StateConfig.mar_smoothing_frames = 15` |
| PERCLOS = T_c / T_T × 100 (Eq. 2) | `state_machine.py` (time-weighted over the window) |

### Deviations from / extensions of the paper (please read)

* **MAD thresholds are an extension.** The paper's threshold rule is a ratio of baseline values (75 % EAR,
  140 % MAR). The robust rule `T_ear = median − k·MAD/0.6745` was requested for this project and is the default
  (`mode="mad"`); the paper's rule is available with `--calib-mode ratio`. Safeguards (sigma floor as a
  *fraction of the driver's own median*, clamp of `T_ear/median` to `[0.55, 0.90]`, `T_mar ≥ 1.40 × median`) keep
  the MAD rule consistent with the paper's operating region. `k_ear = 5` is a design choice, not a paper value.
* **Landmarks:** the paper uses MediaPipe Face Mesh (468 points), not dlib's 68-point model. This code uses
  the Face Mesh vertices that correspond to the classic p1…p6 eye layout and inner-lip points (`config.py`).
  MAR averages **three** upper/lower inner-lip pairs instead of the paper's two, to lower jitter.
* **Fusion is an extension.** The paper evaluates the personalised-threshold method and the CNN method
  *independently* under identical warning logic. Here they are blended (`fuse_probabilities`), with CNN-only
  fallback when geometry is untrusted (head turned) and geometry-only fallback when no weights exist.
* **No pretrained CNN weights are included.** Train them with `train_cnn.py` on the MRL Eye and Yawn datasets
  used in the paper (links in the references). Without weights the system runs geometry-only.
* **Alert levels, window limits and yawn / micro-sleep durations** (`StateConfig`) are not specified in the
  paper; they are conventional engineering defaults and should be validated for your deployment.

## 3. Benchmark comparison

### 3.1 Static global thresholds (legacy) vs. dynamic personalised calibration — results reported by Ersoy et al.

These are the **paper's numbers** (their datasets: MRL Eye, Yawn Dataset, custom data; 70/15/15 split). They are
quoted, not reproduced, by this repository.

| Task | Static global threshold (legacy) | Dynamic personalised baseline (Ersoy et al., 2026) | CNN classifier |
|---|---|---|---|
| Eye state accuracy (Table I) | 91.70 % | 93.23 % (+1.53 pt) | 99.10 % |
| Yawn detection accuracy (Table II) | 95.90 % | 97.23 % (+1.33 pt) | 98.80 % |
| Open-eye class accuracy (Fig. 5) | 89.0 % | 98.9 % | – |
| Closed-eye class accuracy (Fig. 5) | 95.0 % | not stated in text | – |
| Yawn class accuracy, MAR (Fig. 5) | 92.5 % | 98.4 % | – |
| No-yawn class accuracy (Fig. 5) | 98.3 % | 97.3 % | – |

Note on wording: the often-quoted jump **92.5 % → 98.4 %** is the *yawn-class* accuracy of the MAR confusion
matrix. For eye state the overall improvement from personalisation is 91.70 % → 93.23 %, and the strongest eye
result is the CNN (99.10 %). The abstract states a 2–3 % gain from personalisation, while Tables I–II show
+1.53 and +1.33 points.

### 3.2 Calibration logic check on synthetic drivers (this repository)

`python benchmark.py simulation --subjects 200` simulates drivers with different eye / lip geometry. The global
threshold is the paper's ratio applied to the *population* mean baseline (leave-one-out); personalised
thresholds come from `calibration.py` using each driver's first 150 frames (with ~8 % blinks).

| Method (synthetic data, seed 7) | Accuracy | Worst-10 % of drivers |
|---|---|---|
| EAR global | 94.2 % | 83.7 % |
| EAR personalised (ratio rule) | 100.0 % | 100.0 % |
| EAR personalised (median − k·MAD/0.6745) | 100.0 % | 100.0 % |
| MAR global | 84.2 % | 38.1 % |
| MAR personalised (ratio rule) | 99.2 % | 97.4 % |
| MAR personalised (median + k·MAD/0.6745) | 100.0 % | 100.0 % |

**This is a simulation.** The data are generated with person-specific baselines, so near-perfect personalised
scores are expected and are *not* evidence of real-world accuracy. It shows the mechanism (a global cut-off
penalises drivers at the extremes of the population) and verifies that the code implements it.

### 3.3 Speed (CPU, 2 threads, this repository's sandbox)

| Component | Time per frame |
|---|---|
| Eye CNN (2 patches) + mouth CNN, every frame | ≈ 2.9 ms (≈ 60 k parameters each) |
| Same, stride 2 (default) | ≈ 1.6 ms |
| Calibrated scores + state machine | < 0.02 ms |

The MediaPipe landmark stage dominates; measure end-to-end on your machine with
`python benchmark.py video --source 0`. A 30 FPS budget is 33 ms per frame.

## 4. System architecture

```
 camera / video file
        │  BGR frame (640×480)
        ▼
┌──────────────────────┐   face lost → face_found=False, EMA reset, re-acquisition flag
│ landmark_extractor   │──────────────────────────────────────────────────────────────┐
│  MediaPipe Face Mesh │                                                              │
│  EAR, MAR, head ratio│  eye ×2 + mouth crops 64×64 (CLAHE)                          │
└──────────┬───────────┘───────────────────────────┐                                  │
           │ raw EAR / MAR / head ratio            │                                  │
           ▼                                       ▼                                  │
 ┌───────────────────┐                   ┌────────────────────┐                       │
 │ PHASE: CALIBRATING│                   │  cnn_classifier    │                       │
 │ calibration.py    │                   │  Eye CNN / Mouth CNN│                      │
 │ N=150 valid frames│                   │  P(closed), P(yawn)│                       │
 │ median + MAD      │                   └─────────┬──────────┘                       │
 │ → T_ear,T_mar,T_hd│                             │                                  │
 └────────┬──────────┘                             │                                  │
          │ DriverProfile (JSON)                   │                                  │
          ▼                                        ▼                                  │
 ┌──────────────────────────────────────────────────────────────┐                     │
 │ PHASE: TRACKING   smoothing (3-frame median / 15-frame mean)  │                     │
 │  geometric score = sigmoid around T_ear / T_mar (per driver)  │                     │
 │  fused p = w·geometry + (1−w)·CNN   (CNN-only if geometry     │                     │
 │  untrusted; geometry-only if no weights)  → hysteresis        │                     │
 └──────────────────────────────┬───────────────────────────────┘                     │
                                │ eye_closed, yawning, head_off_axis                  │
                                ▼                                                     │
                  ┌───────────────────────────┐                                       │
                  │ state_machine (60 s window)│◄──────────────────────────────────────┘
                  │ N_b, N_s, N_y, PERCLOS     │
                  └─────────────┬─────────────┘
                                ▼
              Level 0 NORMAL │ Level 1 WARNING │ Level 2 CRITICAL
                                ▼
                       main.py HUD / CSV log / video out
```

## 5. Installation and execution

Requires Python 3.10+ and a webcam or a video file.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # numpy, opencv-python, mediapipe, torch
```

**Face model.** Recent MediaPipe releases need `models/face_landmarker.task`. It is downloaded automatically on first
run; if you are offline, fetch it manually from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
and save it to `models/face_landmarker.task`.

**Run the monitor**

```bash
python main.py                                  # webcam, calibrate then track
python main.py --source drive.mp4               # video file
python main.py --profile profiles/me.json       # load the profile if it exists, otherwise calibrate + save
python main.py --recalibrate --profile profiles/me.json
python main.py --no-cnn                         # geometry-only (personalised-threshold branch)
python main.py --calib-mode ratio               # paper's 75 % / 140 % rule instead of MAD
python main.py --source drive.mp4 --no-display --output out.mp4 --log-csv log.csv
```

**Calibrating.** For the first ~5 s of valid frames (progress bar on screen) sit in your normal driving posture,
look at the camera, eyes open, mouth closed, face still. Normal blinks are tolerated (outlier rejection); if
the window is unusable (eyes closed, large motion) it restarts and says why. Keys: `r` recalibrate, `s` save
profile, `q` / `Esc` quit.

**Train the CNNs (optional but recommended)**

```bash
# folders: data/eyes/{open,closed}   data/yawn/{no_yawn,yawn}
python train_cnn.py --task eye   --data-dir data/eyes --epochs 15
python train_cnn.py --task mouth --data-dir data/yawn --epochs 20
```

Weights are written to `models/eye_cnn.pt` and `models/mouth_cnn.pt`; the test accuracy and confusion
matrix of the best-validation checkpoint are printed.

**Tests and benchmarks**

```bash
python -m unittest discover -s tests -v         # 29 tests, no camera needed
python benchmark.py simulation --subjects 200
python benchmark.py latency
python benchmark.py video --source drive.mp4 --frames 600
```

## 6. Project layout

```
config.py              all runtime parameters (frozen dataclasses; no EAR/MAR decision constants)
calibration.py         median/MAD calibration, DriverProfile, online Calibrator
landmark_extractor.py  Face Mesh → EAR, MAR, head ratio, CLAHE crops; loss/re-acquisition handling
cnn_classifier.py      StateCNN (~61 k params), inference wrapper, fusion, hysteresis
state_machine.py       60 s window: N_b, N_s, N_y, PERCLOS, alert levels with hysteresis
main.py                pipeline, HUD (FPS, phase, thresholds, live EAR/MAR graphs, banners), CLI
train_cnn.py           training / evaluation of the two CNNs
benchmark.py           simulation, latency and end-to-end benchmarks
tests/test_pipeline.py unit tests
```

### Key formulas

```
MAD(x)    = median(|x − median(x)|),   σ = MAD / 0.6745
T_ear     = median(EAR) − k_ear · max(σ_ear, 0.05·median)      clamped to [0.55, 0.90]·median
T_mar     = median(MAR) + k_mar · max(σ_mar, 0.15·median)      and ≥ 1.40·median
EAR       = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖)
MAR       = mean(3 inner-lip vertical distances) / mouth width
PERCLOS   = time eyes closed / window time   (60 s, time-weighted)
```

## 7. Limitations and honest caveats

* The code was verified with unit tests on synthetic landmarks and with synthetic CNN training data. **End-to-end
  accuracy on real faces has not been measured here** and the CNN weights are not provided; reproduce the paper's
  accuracies by training on the cited datasets and evaluating on held-out data.
* MAR on inner-lip points is near zero for a closed mouth, so the personalised yawn threshold is small; talking can
  exceed it briefly. The 1.5 s minimum duration, the 15-frame mean and CNN fusion are what suppress false yawns —
  validate on your own data.
* Head-off-axis uses a 2-D proxy (nose-tip to eye-line ratio, direction-agnostic); it is not a full pose estimate.
* Alert limits (PERCLOS 15 % / 30 %, micro-sleep ≥ 0.5 s, etc.) are engineering defaults, not clinically validated.
* This is a research prototype and not a certified safety device.

## 8. References

1. G. Ersoy, M. A. Tatar, E. Tonbul, S. Kırbız, "Improving Driver Drowsiness Detection via Personalized EAR/MAR
   Thresholds and CNN-Based Classification," arXiv:2604.22479v1, 2026. <https://arxiv.org/html/2604.22479v1?hl=en-GB>
2. T. Soukupová and J. Čech, "Real-time eye blink detection using facial landmarks," CMP, CTU Prague, 2016.
3. MRL Eye Dataset (Kaggle, A. Shingha) and Yawn Dataset (Kaggle, D. Vazquez CIC), as used in [1].
4. Google MediaPipe Face Landmarker / Face Mesh.
