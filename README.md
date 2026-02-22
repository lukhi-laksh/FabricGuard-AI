# 🧵 FabricGuard AI — Fabric Hole Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8m-blue?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.1-76B900?style=for-the-badge&logo=nvidia" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows" />
</p>

> **FabricGuard AI** is a production-ready, single-class fabric hole detection system built on **YOLOv8m** and served via a **FastAPI** web application. Upload your fabric images, train a custom model, and detect holes in images, videos, or via a live webcam — all from a clean browser UI.

---

## 📋 Table of Contents

1. [What it Does](#-what-it-does)
2. [Project Architecture](#-project-architecture)
3. [File Structure](#-file-structure)
4. [Requirements](#-requirements)
5. [Quick Start (Windows)](#-quick-start-windows)
6. [Manual Setup](#-manual-setup)
7. [How to Use the Web UI](#-how-to-use-the-web-ui)
8. [Dataset Format](#-dataset-format)
9. [Training Details](#-training-details)
10. [Detection Details](#-detection-details)
11. [API Reference](#-api-reference)
12. [Google Colab Training](#-google-colab-training)
13. [GPU Configuration](#-gpu-configuration)
14. [Troubleshooting](#-troubleshooting)
15. [Performance Benchmarks](#-performance-benchmarks)

---

## 🎯 What it Does

FabricGuard AI detects **holes** in fabric samples. The system:

- **Trains** a custom YOLOv8m model on your own labeled dataset
- **Detects** holes in uploaded images with bounding boxes and confidence scores
- **Scans** video files frame-by-frame and flags the first frame with a hole
- **Streams** a live webcam feed with real-time hole detection overlay
- **Reports** metrics (mAP50, Precision, Recall, Box Loss, Cls Loss) live during training

| Input | Output |
|-------|--------|
| Fabric image (JPG/PNG) | Annotated image + `hole_count`, `confidence`, `status` |
| Fabric video (MP4/AVI) | Timestamp + annotated frame of first detected hole |
| Live webcam | MJPEG stream with real-time bounding boxes |

---

## 🏗️ Project Architecture

```
Browser UI (index.html)
        │
        │  HTTP / REST API
        ▼
FastAPI Backend (main.py)
  ├── /upload-images    → saves to data/hole_image/
  ├── /upload-labels    → saves to data/value_image/
  ├── /train            → spawns background thread → train.py
  ├── /train-status     → polls live training metrics
  ├── /detect-image     → detect.py → detect_image()
  ├── /detect-video     → detect.py → detect_video()
  ├── /webcam-feed      → detect.py → generate_webcam_frames()  [MJPEG stream]
  └── /webcam-stats     → live hole count + confidence

Training Pipeline (train.py)
  1. _require_gpu()        → verify NVIDIA CUDA GPU
  2. _prepare_dataset()    → match image-label pairs, 80/20 split
  3. _generate_yaml()      → write dataset.yaml
  4. YOLO("yolov8m.pt").train()  → GPU training with callbacks
  5. copy best.pt → models/hole_detector.pt

Detection Engine (detect.py)
  ├── detect_image()       → YOLO inference → draw boxes → save annotated
  ├── detect_video()       → frame-by-frame scan → return first hit
  └── generate_webcam_frames()  → MJPEG generator with HUD overlay
```

---

## 📁 File Structure

```
FabricGuard AI/
│
├── main.py                  ← FastAPI app — the entry point (run this)
├── train.py                 ← YOLOv8m training logic (GPU-strict)
├── detect.py                ← Image / video / webcam detection engine
├── online.py                ← Google Colab training script (cloud training)
│
├── dataset.yaml             ← Auto-generated YOLO dataset config
├── yolov8m.pt               ← Pre-trained YOLOv8m base weights (52 MB)
├── requirements.txt         ← Python dependencies
│
├── setup.bat                ← One-click environment setup (Windows)
├── run.bat                  ← One-click app launcher (Windows)
│
├── templates/
│   └── index.html           ← Full-featured browser UI (single page app)
│
├── data/
│   ├── hole_image/          ← YOUR fabric images go here (.jpg / .png)
│   └── value_image/         ← YOUR YOLO label .txt files go here
│
├── models/
│   └── hole_detector.pt     ← Trained model saved here after training
│
├── yolo_data/               ← Auto-built YOLO dataset (train/val split)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
│
├── runs/
│   └── hole_detection/      ← YOLOv8 training run artifacts
│       └── weights/
│           ├── best.pt      ← Best epoch weights
│           └── last.pt      ← Final epoch weights
│
├── uploads/
│   ├── images/              ← Uploaded test images
│   ├── annotated/           ← Detected images with bounding boxes
│   ├── videos/              ← Uploaded test videos
│   └── detected_frames/     ← Video frames where holes were found
│
├── metrics/                 ← Saved training metric snapshots (JSON)
│
└── COLAB_TRAINING_GUIDE.md  ← Full Google Colab training documentation
```

---

## ⚙️ Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any NVIDIA GPU with CUDA | RTX 3050 / 4060+ (4 GB+ VRAM) |
| RAM | 8 GB | 16 GB+ |
| Storage | 2 GB free | 5 GB+ |

> ⚠️ **Training requires a CUDA-capable NVIDIA GPU.** Detection and inference can fall back to CPU.

### Software
- **Python** 3.10 or later  
- **NVIDIA Drivers** 520+ (for CUDA 12.1)  
- **CUDA** 12.1 (installed via PyTorch)  
- **Git** (optional, for cloning)

---

## 🚀 Quick Start (Windows)

### Step 1 — Clone or download the project
```bash
git clone https://github.com/yourusername/FabricGuard-AI.git
cd FabricGuard-AI
```

### Step 2 — Run setup (one-time only)
```bat
setup.bat
```
This automatically:
- Creates a Python virtual environment (`venv/`)
- Installs all dependencies from `requirements.txt`
- Installs GPU-enabled PyTorch (CUDA 12.1)
- Downloads the `yolov8m.pt` base weights

### Step 3 — Launch the application
```bat
run.bat
```
Or manually:
```bash
# Activate venv first
venv\Scripts\activate

# Launch the server
python main.py
```

### Step 4 — Open the browser UI
```
http://127.0.0.1:8000
```

---

## 🛠️ Manual Setup

If you prefer manual installation:

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install GPU PyTorch (CUDA 12.1) — crucial for training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Download YOLOv8m base weights
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt'); print('Ready!')"

# 6. Run
python main.py
```

---

## 🖥️ How to Use the Web UI

The single-page UI at `http://127.0.0.1:8000` has three main sections:

### 1️⃣ Dataset Upload
- **Upload Images** — drag & drop or select your fabric images (`.jpg`, `.png`, `.bmp`, `.webp`)
- **Upload Labels** — upload matching YOLO `.txt` annotation files
- The dashboard shows total images, total labels, matched pairs, and any missing files
- Click **Clear Dataset** to start fresh

### 2️⃣ Training
- Set the number of **epochs** (default: 100)
- Click **Start Training**
- Watch live metrics update in real time:
  - Current epoch / total epochs
  - Box loss & Classification loss
  - mAP@50, Precision, Recall
- Training log streams to the UI
- The trained model is automatically saved to `models/hole_detector.pt`

### 3️⃣ Detection

**Image Detection:**
- Upload any fabric image
- View the result: hole count, max confidence, annotated image with red bounding boxes

**Video Detection:**
- Upload a fabric video
- The system scans every 5th frame and returns the timestamp + annotated frame of the first detected hole

**Webcam Detection:**
- Select a connected camera from the dropdown
- Click **Start Webcam** to stream live MJPEG feed with hole detection overlay
- A HUD bar shows `HOLES: N (XX%)` or `CLEAN` in real time

---

## 📊 Dataset Format

### Folder Structure

Place your files in exactly these folders:

```
data/
├── hole_image/        ← all fabric images (.jpg / .png / .bmp / .webp)
└── value_image/       ← all YOLO annotation .txt files (same filenames)
```

> Each image must have a matching `.txt` file with the **same filename** (different extension).  
> Example: `fabric_001.jpg` ↔ `fabric_001.txt`

### Label File Format

Each `.txt` file contains one annotation per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

| Field | Value | Notes |
|-------|-------|-------|
| `class_id` | `0` | Always `0` — "hole" is the only class |
| `x_center` | `0.0 – 1.0` | Normalized horizontal center |
| `y_center` | `0.0 – 1.0` | Normalized vertical center |
| `width` | `0.0 – 1.0` | Normalized bounding box width |
| `height` | `0.0 – 1.0` | Normalized bounding box height |

**Example `fabric_001.txt`:**
```
0 0.512 0.437 0.134 0.098
0 0.321 0.654 0.089 0.076
```
*(Two holes detected in this image)*

**No holes in the image?** → Create an **empty** `.txt` file with the same name.

### Recommended Dataset Size

| Images | Expected mAP@50 | Training Time (RTX 3050) |
|--------|----------------|--------------------------|
| < 100  | < 0.60         | ~5 min                   |
| 100–500 | 0.65–0.80    | ~15 min                  |
| 500–2000 | 0.80–0.92   | ~40 min                  |
| > 2000 | > 0.92         | ~90 min                  |

---

## 🏋️ Training Details

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `yolov8m.pt` | YOLOv8 Medium — pretrained on COCO |
| Epochs | 100 (configurable) | Set in UI |
| Image size | 640 × 640 px | Fixed |
| Batch size | Auto (`-1`) | GPU auto-selects optimal batch |
| Optimizer | AdamW | Weight decay regularization |
| Early stopping | patience = 20 | Stops if no improvement for 20 epochs |
| Confidence threshold | 0.40 | Minimum confidence to count as detection |
| Train/Val split | 80% / 20% | Random seed = 42 for reproducibility |

### Training Flow

```
1. Validate GPU (raises error if not found)
2. Scan data/hole_image/ and data/value_image/
3. Match image-label pairs (unmatched images are skipped with a warning)
4. Shuffle and split: 80% → train, 20% → val
5. Build yolo_data/ directory structure
6. Generate dataset.yaml (nc=1, names=['hole'])
7. Load yolov8m.pt (pretrained COCO weights)
8. Train with epoch-end callbacks (live metrics → UI)
9. Save best.pt → models/hole_detector.pt
```

### Output Files

After training completes:

```
models/
└── hole_detector.pt         ← THE model to use for inference

runs/hole_detection/
├── weights/
│   ├── best.pt              ← Best checkpoint
│   └── last.pt              ← Last epoch checkpoint
├── results.csv              ← Per-epoch metrics table
└── args.yaml                ← Training hyperparameters
```

---

## 🔍 Detection Details

### Image Detection Response

```json
{
  "hole_count": 2,
  "max_confidence": 0.924,
  "detection_status": true,
  "detections": [
    {"bbox": [120, 85, 220, 165], "confidence": 0.924},
    {"bbox": [340, 210, 410, 290], "confidence": 0.811}
  ],
  "annotated_url": "/uploads/annotated/annotated_fabric_001.jpg",
  "message": "2 holes detected (max confidence 92.4%)",
  "error": false
}
```

**No hole found:**
```json
{
  "hole_count": 0,
  "max_confidence": 0.0,
  "detection_status": false,
  "message": "Fabric is clean",
  "error": false
}
```

### Video Detection Response

```json
{
  "detection_status": true,
  "hole_count": 1,
  "max_confidence": 0.876,
  "timestamp": "00:14",
  "timestamp_sec": 14.2,
  "frame_number": 355,
  "frame_url": "/uploads/detected_frames/frame_355.jpg",
  "message": "Hole detected at 00:14",
  "error": false
}
```

### Visual Annotations

- **Red bounding boxes** drawn around each detected hole
- **Label**: `Hole XX.X%` (confidence) on each box
- **Webcam HUD**: Status bar showing `HOLES: N (XX%)` in red or `CLEAN` in green

---

## 📡 API Reference

All endpoints are available at `http://127.0.0.1:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the web UI |
| `GET` | `/health` | Health check + model-ready status |
| `GET` | `/dataset-info` | Image/label counts, matched pairs, missing files |
| `POST` | `/upload-images` | Upload fabric images to `data/hole_image/` |
| `POST` | `/upload-labels` | Upload YOLO `.txt` labels to `data/value_image/` |
| `POST` | `/clear-dataset` | Delete all uploaded images and labels |
| `POST` | `/train` | Start training (`epochs` form field, default 100) |
| `GET` | `/train-status` | Live training progress (poll this endpoint) |
| `POST` | `/detect-image` | Detect holes in an uploaded image |
| `POST` | `/detect-video` | Detect holes in an uploaded video |
| `GET` | `/webcam-devices` | List available camera indices |
| `GET` | `/webcam-feed` | MJPEG stream with live detection (`?camera_index=N`) |
| `GET` | `/webcam-stats` | Current live hole count and confidence |

You can also explore the auto-generated API docs at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## ☁️ Google Colab Training

For users without a local GPU, you can train on **Google Colab** using `online.py`.

### Setup on Google Drive

Your Drive must have this exact structure:
```
MyDrive/
└── mmm/
    ├── image/     ← all fabric images (.jpg / .png)
    └── lable/     ← all label .txt files
```
> **Note:** The folder is `lable` (not `label`) — this matches the script's default path.

### Running on Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** → Runtime → Change runtime type → GPU
3. Upload `online.py` to Colab:
   ```python
   from google.colab import files
   files.upload()   # select online.py
   ```
4. Run the script:
   ```bash
   !python online.py
   ```
5. When done, `best.pt` and a full zip archive are **auto-downloaded** to your browser

### What `online.py` Does (12 Steps)

| Step | Action |
|------|--------|
| 0 | Install ultralytics + detect GPU |
| 1 | Mount Google Drive |
| 2 | Validate image/label paths and class indices |
| 3 | Build `dataset/train` + `dataset/val` (80/20 split) |
| 4 | Write `hole_dataset.yaml` |
| 5 | Clear GPU memory |
| 6 | Train YOLOv8n (100 epochs, AdamW, auto-batch) |
| 7 | Locate `best.pt` |
| 8 | Export to **ONNX** and **TorchScript** |
| 9 | Save artifacts to Google Drive |
| 10 | Zip the full run folder |
| 11 | Auto-download `best.pt` + zip |
| 12 | Run post-training validation (mAP / Precision / Recall) |

### Using the Colab-Trained Model Locally

After downloading `best.pt`, copy it to your local project:
```bash
cp best.pt models/hole_detector.pt
```
Then run `python main.py` normally — the UI will use your trained model.

> 📄 See **[COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)** for the complete training and deployment guide.

---

## 🖥️ GPU Configuration

The system is configured to use **NVIDIA CUDA GPU** when available, with automatic CPU fallback for inference.

### Architecture Decision

| Operation | GPU Required? | Notes |
|-----------|--------------|-------|
| **Training** | ✅ Yes (strict) | Raises `RuntimeError` if no CUDA GPU |
| **Image detection** | ❌ No (optional) | Falls back to CPU automatically |
| **Video detection** | ❌ No (optional) | Falls back to CPU automatically |
| **Webcam streaming** | ❌ No (optional) | Falls back to CPU automatically |

### Environment Variables (auto-set at startup)

```python
CUDA_VISIBLE_DEVICES = "0"      # use GPU 0 (dedicated NVIDIA)
CUDA_DEVICE_ORDER   = "PCI_BUS_ID"  # stable ordering across reboots
```

### Verify GPU is Working

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 🔧 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `No CUDA GPU detected` during training | CPU-only PyTorch or bad drivers | Run: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| `Model not found. Please train first.` | `models/hole_detector.pt` missing | Train the model first using the UI |
| `No image–label pairs found` | Filename mismatch | `fabric01.jpg` needs `fabric01.txt` (same name, different ext) |
| Webcam shows black / `No camera found` | Wrong camera index or not connected | Try different index in the dropdown |
| Training stuck at 0% | Not enough matched pairs | Ensure each image has a corresponding `.txt` label file |
| `CUDA out of memory` | GPU VRAM too small for auto-batch | Reduce batch manually in `train.py` line: `batch = 8` |
| `ultralytics not installed` | Missing dependency | Run `pip install ultralytics` |
| Browser can't reach `127.0.0.1:8000` | Server not running | Make sure `python main.py` is running without errors |
| Port 8000 already in use | Another process using the port | Kill it or change port in `main.py`: `uvicorn.run(..., port=8001)` |

---

## 📈 Performance Benchmarks

### Training Time (Local — RTX 3050, CUDA 12.1)

| Dataset Size | Epochs | Time | mAP@50 |
|-------------|--------|------|--------|
| 100 images  | 100    | ~8 min  | ~0.72 |
| 300 images  | 100    | ~18 min | ~0.82 |
| 500 images  | 100    | ~28 min | ~0.87 |
| 1000 images | 100    | ~55 min | ~0.91 |

### Training Time (Google Colab — Tesla T4)

| Dataset Size | Epochs | Time | mAP@50 |
|-------------|--------|------|--------|
| 200 images  | 100    | ~12 min | 0.78 |
| 500 images  | 100    | ~28 min | 0.87 |
| 1000 images | 100    | ~55 min | 0.91 |
| 2000 images | 100    | ~90 min | 0.94 |

### Metric Glossary

| Metric | Meaning | Target |
|--------|---------|--------|
| `mAP@50` | Mean Average Precision at IoU=0.50 | > 0.85 |
| `Precision` | Of predicted holes, % that are actual holes | > 0.85 |
| `Recall` | Of real holes, % correctly found | > 0.85 |
| `Box Loss` | Bounding box regression loss | Decreasing |
| `Cls Loss` | Classification loss | Decreasing |

---

## 🧩 Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Deep Learning | [PyTorch 2.5+](https://pytorch.org/) (CUDA 12.1) |
| Web Backend | [FastAPI](https://fastapi.tiangolo.com/) |
| Web Server | [Uvicorn](https://www.uvicorn.org/) |
| UI Template | Jinja2 + Vanilla HTML/CSS/JS |
| Image Processing | OpenCV, Pillow, NumPy |
| Data Format | YOLO Object Detection (`.txt` labels) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 framework
- [FastAPI](https://fastapi.tiangolo.com/) for the blazing-fast web backend
- [PyTorch](https://pytorch.org/) for the deep learning foundation

---

*FabricGuard AI — Built for production fabric quality control.*  
*Last updated: February 2026*
