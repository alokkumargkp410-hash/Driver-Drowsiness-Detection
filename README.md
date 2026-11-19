# Personalised Driver Drowsiness Detection

A real-time Computer Vision system for detecting driver drowsiness using MediaPipe Face Mesh, personalised EAR/MAR calibration, optional CNN-based eye and mouth classification, and temporal analysis.

## 1. Abstract

Driver drowsiness is an important factor affecting road safety. This project presents a real-time driver drowsiness detection system that monitors facial features and identifies signs of fatigue.

The system uses MediaPipe Face Mesh to extract facial landmarks and calculate Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR). Instead of relying entirely on fixed thresholds, it performs personalised calibration to determine driver-specific thresholds using median and Median Absolute Deviation (MAD) statistics.

Optional Convolutional Neural Networks (CNNs) classify eye states (open/closed) and mouth states (yawning/neutral). The extracted information is processed through a temporal state machine that generates normal, warning, and critical alerts.

## 2. Objectives

- Detect driver drowsiness in real time.
- Extract facial landmarks using MediaPipe.
- Calculate EAR and MAR for eye and mouth monitoring.
- Implement personalised threshold calibration.
- Integrate CNN-based eye and mouth classification.
- Reduce false alerts through temporal analysis.
- Generate warning and critical alerts.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image and video processing |
| MediaPipe | Facial landmark detection |
| PyTorch | CNN implementation |
| NumPy | Numerical operations |
| Matplotlib | Visualisation and analysis |

## 4. System Methodology

The proposed system consists of the following stages:

### 4.1 Video Input
The system captures real-time video through a webcam or accepts a recorded video.

### 4.2 Facial Landmark Detection
MediaPipe Face Mesh identifies facial landmarks around the eyes, mouth, and other facial regions.

### 4.3 Feature Extraction
The system calculates:
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head-position proxy

### 4.4 Personalised Calibration
During the initial calibration period, the system collects approximately 150 valid frames and calculates individual baseline values.

Median and MAD statistics are used to determine personalised EAR and MAR thresholds.

### 4.5 CNN Classification
Optional CNN models classify:
- Eye state: Open or Closed.
- Mouth state: Neutral or Yawning.

The CNN outputs are combined with geometric information through a fusion mechanism.

**Note:** Pretrained CNN weights are not included in this repository. The models can be trained using the provided training script and suitable datasets.

### 4.6 Temporal Analysis
The state machine analyses information over a 60-second sliding window.

It monitors:
- Blink frequency
- Micro-sleep duration
- Yawning frequency
- PERCLOS

### 4.7 Alert Generation
The system generates three levels:

- Level 0: Normal
- Level 1: Warning
- Level 2: Critical

## 5. System Architecture

```text
Camera / Video Input
        |
        v
MediaPipe Face Mesh
        |
        v
Facial Landmark Extraction
        |
        +--------------------+
        |                    |
        v                    v
 EAR / MAR Features     CNN Classification
        |                    |
        v                    v
Personalised Calibration    CNN Probabilities
        |                    |
        +---------+----------+
                  |
                  v
             Feature Fusion
                  |
                  v
           Temporal Analysis
                  |
                  v
            Alert Generation
                  |
                  v
          Normal / Warning /
              Critical
```

## 6. Installation

### Requirements

- Python 3.10 or newer
- Webcam or video input
- Windows, Linux, or macOS

### Step 1: Clone the repository

```bash
git clone https://github.com/alokkumargkp410-hash/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

### Step 2: Create a virtual environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## 7. Execution

### Run using webcam

```bash
python main.py
```

### Run using a video file

```bash
python main.py --source drive.mp4
```

### Run without CNN classification

```bash
python main.py --no-cnn
```

### Recalibrate the system

```bash
python main.py --recalibrate --profile profiles/me.json
```

### Use ratio-based calibration

```bash
python main.py --calib-mode ratio
```

The system performs an initial calibration before tracking. During calibration, the driver should maintain a normal posture, look towards the camera, and keep the eyes open.

## 8. CNN Model Training

The project provides a training script for the eye and mouth CNN models.

Expected dataset structure:

```text
data/
    eyes/
        open/
        closed/

    yawn/
        no_yawn/
        yawn/
```

Train the eye classification model:

```bash
python train_cnn.py --task eye --data-dir data/eyes --epochs 15
```

Train the mouth classification model:

```bash
python train_cnn.py --task mouth --data-dir data/yawn --epochs 20
```

The trained model weights are saved in:

```text
models/
    eye_cnn.pt
    mouth_cnn.pt
```

The model weights are not included in the current repository and must be generated separately.

## 9. Testing and Benchmarking

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

Run simulation benchmark:

```bash
python benchmark.py simulation --subjects 200
```

Run latency benchmark:

```bash
python benchmark.py latency
```

Run video benchmark:

```bash
python benchmark.py video --source drive.mp4 --frames 600
```

### Results and Evaluation

The repository includes synthetic simulation benchmarks for checking personalised calibration behaviour.

These results are not real-world accuracy measurements.

The research paper's reported accuracy values are reference results and should not be interpreted as independently achieved results by this implementation.

End-to-end evaluation on real-world driving videos is still required.

## 10. Project Structure

```text
Driver-Drowsiness-Detection/
│
├── models/
│   └── .gitkeep
│
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── main.py
├── config.py
├── calibration.py
├── landmark_extractor.py
├── cnn_classifier.py
├── state_machine.py
├── train_cnn.py
├── benchmark.py
└── requirements.txt
```

## 11. Limitations

- CNN model weights are not provided and require separate training.
- Real-world accuracy has not yet been established.
- Facial landmark detection can be affected by lighting and head movement.
- Talking may sometimes be confused with yawning.
- The system is a research prototype and is not a certified vehicle safety device.

## 12. Future Enhancements

- Improve CNN classification using larger datasets.
- Evaluate performance under different lighting conditions.
- Improve head-pose estimation.
- Reduce false-positive alerts.
- Perform real-world testing under different driving conditions.

## 13. References

1. G. Ersoy, M. A. Tatar, E. Tonbul, S. Kırbız, "Improving Driver Drowsiness Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification," arXiv:2604.22479v1, 2026.
2. T. Soukupová and J. Čech, "Real-time eye blink detection using facial landmarks," 2016.
3. MRL Eye Dataset.
4. Yawn Dataset.
5. Google MediaPipe Face Landmarker documentation.

---

**Project:** Personalised Driver Drowsiness Detection

**Domain:** Computer Vision and Deep Learning

**Language:** Python
